// Original row-graph scheduling pass, built into the pinned IREE compiler.
#include <algorithm>
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {
class LabDialect final : public Dialect {
 public:
  explicit LabDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(),context,TypeID::get<LabDialect>()) {
    allowUnknownOperations();
  }
  static StringRef getDialectNamespace() { return "lab"; }
};

struct SchedulePass final : PassWrapper<SchedulePass,OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SchedulePass)
  StringRef getArgument() const override { return "lab-schedule-row-mlp"; }
  StringRef getDescription() const override { return "Lower a verified row-local residual MLP graph to a bounded GPU task schedule"; }
  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<LabDialect>(); }
  void runOnOperation() override {
    SmallVector<Operation*> graphs;
    getOperation()->walk([&](Operation *op) {
      if(op->getName().getStringRef()=="lab.graph") graphs.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    for(Operation *op : graphs) {
      auto rows=op->getAttrOfType<IntegerAttr>("rows");
      auto channels=op->getAttrOfType<IntegerAttr>("channels");
      auto hidden=op->getAttrOfType<IntegerAttr>("hidden");
      auto kind=op->getAttrOfType<StringAttr>("kind");
      auto dtype=op->getAttrOfType<StringAttr>("dtype");
      auto scope=op->getAttrOfType<StringAttr>("dependency_scope");
      if(!rows || !channels || !hidden || !kind || !dtype || !scope ||
         rows.getInt()<1 || channels.getInt()<1 || hidden.getInt()<1 ||
         kind.getValue()!="residual_mlp" || dtype.getValue()!="f32" || scope.getValue()!="independent_rows" ||
         op->getNumOperands()!=0 || op->getNumResults()!=0 || op->getNumRegions()!=0) {
        op->emitError("requires a positive-shape f32 residual_mlp graph with proven independent rows");
        signalPassFailure(); return;
      }
      bool supported=channels.getInt()<=256 && hidden.getInt()<=256;
      // Explicit conservative initial policy, calibrated only on development shapes.
      // The large-row scalar path loses to conventional GEMM/graph execution.
      bool persistent=supported && rows.getInt()<=64 && channels.getInt()<=64 && hidden.getInt()<=128;
      rewriter.setInsertionPoint(op);
      OperationState state(op->getLoc(),"lab.schedule");
      state.addAttributes(op->getAttrs());
      state.attributes.set("strategy",rewriter.getStringAttr(persistent?"persistent":"conventional"));
      state.attributes.set("workers",rewriter.getI64IntegerAttr(std::min<int64_t>(16,rows.getInt())));
      state.attributes.set("shared_bytes",rewriter.getI64IntegerAttr(persistent?4*(channels.getInt()+hidden.getInt()):0));
      state.attributes.set("synchronization",rewriter.getStringAttr(persistent?"cta_only":"kernel_boundaries"));
      state.attributes.set("stages",rewriter.getArrayAttr({rewriter.getStringAttr("projection_relu"),rewriter.getStringAttr("projection_residual_relu")}));
      state.attributes.set("reason",rewriter.getStringAttr(persistent?"bounded row graph; low launch count":"resource/shape policy selects conventional fallback"));
      rewriter.create(state);
      rewriter.eraseOp(op);
    }
  }
};

struct LabOptions { void bindOptions(OptionsBinder&) {} };
struct LabSession : PluginSession<LabSession,LabOptions> {
  static void registerPasses() { PassRegistration<SchedulePass>(); }
  void onRegisterDialects(DialectRegistry &registry) override { registry.insert<LabDialect>(); }
  LogicalResult onActivate() override { context->getOrLoadDialect<LabDialect>(); return success(); }
};
}
IREE_DEFINE_COMPILER_OPTION_FLAGS(LabOptions);
extern "C" bool iree_register_compiler_plugin_lab(PluginRegistrar *registrar) {
  registrar->registerPlugin<LabSession>("lab");return true;
}
