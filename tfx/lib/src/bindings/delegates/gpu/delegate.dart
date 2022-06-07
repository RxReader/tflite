import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:tfx/src/bindings/c/c_api.dart';
import 'package:tfx/src/bindings/dylib.dart' show tfliteGpuLib;

/// [tensorflow#gpu_delegate](https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/lite/delegates/gpu/delegate.h)

/// Encapsulated compilation/runtime tradeoffs.
class TfLiteGpuInferenceUsage {
  const TfLiteGpuInferenceUsage._();

  /// Delegate will be used only once, therefore, bootstrap/init time should
  /// be taken into account.
  static const int TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER = 0;

  /// Prefer maximizing the throughput. Same delegate will be used repeatedly on
  /// multiple inputs.
  static const int TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED = 1;
}

class TfLiteGpuInferencePriority {
  const TfLiteGpuInferencePriority._();

  /// AUTO priority is needed when a single priority is the most important
  /// factor. For example,
  /// priority1 = MIN_LATENCY would result in the configuration that achieves
  /// maximum performance.
  static const int TFLITE_GPU_INFERENCE_PRIORITY_AUTO = 0;
  static const int TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION = 1;
  static const int TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY = 2;
  static const int TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE = 3;
}

/// Used to toggle experimental flags used in the delegate. Note that this is a
/// bitmask, so the values should be 1, 2, 4, 8, ...etc.
class TfLiteGpuExperimentalFlags {
  const TfLiteGpuExperimentalFlags._();

  static const int TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE = 0;

  /// Enables inference on quantized models with the delegate.
  /// NOTE: This is enabled in TfLiteGpuDelegateOptionsV2Default.
  static const int TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT = 1 << 0;

  /// Enforces execution with the provided backend.
  static const int TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY = 1 << 1;
  static const int TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY = 1 << 2;

  /// Enable serialization of GPU kernels & model data. Speeds up initilization
  /// at the cost of space on disk.
  /// Delegate performs serialization the first time it is applied with a new
  /// model or inference params. Later initializations are fast.
  /// ModifyGraphWithDelegate will fail if data cannot be serialized.
  ///
  /// NOTE: User also needs to set serialization_dir & model_token in
  /// TfLiteGpuDelegateOptionsV2.
  /// Currently works only if CL backend is used.
  static const int TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION = 1 << 3;

  static int calculateBitmask(List<int> flags) {
    int bitmask = 0;
    for (final int flag in flags) {
      bitmask |= flag;
    }
    return bitmask;
  }
}

/// IMPORTANT: Always use TfLiteGpuDelegateOptionsV2Default() method to create
/// new instance of TfLiteGpuDelegateOptionsV2, otherwise every new added option
/// may break inference.
class TfLiteGpuDelegateOptionsV2 extends Struct {
  /// When set to zero, computations are carried out in maximal possible
  /// precision. Otherwise, the GPU may quantify tensors, downcast values,
  /// process in FP16 to increase performance. For most models precision loss is
  /// warranted.
  /// [OBSOLETE]: to be removed
  @Int32()
  external int is_precision_loss_allowed;

  /// Preference is defined in TfLiteGpuInferenceUsage.
  @Int32()
  external int inference_preference;

  /// Ordered priorities provide better control over desired semantics,
  /// where priority(n) is more important than priority(n+1), therefore,
  /// each time inference engine needs to make a decision, it uses
  /// ordered priorities to do so.
  /// For example:
  ///   MAX_PRECISION at priority1 would not allow to decrease precision,
  ///   but moving it to priority2 or priority3 would result in F16 calculation.
  ///
  /// Priority is defined in TfLiteGpuInferencePriority.
  /// AUTO priority can only be used when higher priorities are fully specified.
  /// For example:
  ///   VALID:   priority1 = MIN_LATENCY, priority2 = AUTO, priority3 = AUTO
  ///   VALID:   priority1 = MIN_LATENCY, priority2 = MAX_PRECISION,
  ///            priority3 = AUTO
  ///   INVALID: priority1 = AUTO, priority2 = MIN_LATENCY, priority3 = AUTO
  ///   INVALID: priority1 = MIN_LATENCY, priority2 = AUTO,
  ///            priority3 = MAX_PRECISION
  /// Invalid priorities will result in error.
  @Int32()
  external int inference_priority1;
  @Int32()
  external int inference_priority2;
  @Int32()
  external int inference_priority3;

  /// Bitmask flags. See the comments in TfLiteGpuExperimentalFlags.
  @Int64()
  external int experimental_flags;

  /// A graph could have multiple partitions that can be delegated to the GPU.
  /// This limits the maximum number of partitions to be delegated. By default,
  /// it's set to 1 in TfLiteGpuDelegateOptionsV2Default().
  @Int32()
  external int max_delegated_partitions;

  /// The nul-terminated directory to use for serialization.
  /// Whether serialization actually happens or not is dependent on backend used
  /// and validity of this directory.
  /// Set to nullptr in TfLiteGpuDelegateOptionsV2Default(), which implies the
  /// delegate will not try serialization.
  ///
  /// NOTE: Users should ensure that this directory is private to the app to
  /// avoid data access issues.
  external Pointer<Utf8> serialization_dir;

  /// The unique nul-terminated token string that acts as a 'namespace' for
  /// all serialization entries.
  /// Should be unique to a particular model (graph & constants).
  /// For an example of how to generate this from a TFLite model, see
  /// StrFingerprint() in lite/delegates/serialization.h.
  ///
  /// Set to nullptr in TfLiteGpuDelegateOptionsV2Default(), which implies the
  /// delegate will not try serialization.
  external Pointer<Utf8> model_token;

  static Pointer<TfLiteGpuDelegateOptionsV2> allocate(
    int /*0/1*/ is_precision_loss_allowed,
    int /*TfLiteGpuInferenceUsage*/ inference_preference,
    int /*TfLiteGpuInferencePriority*/ inference_priority1,
    int /*TfLiteGpuInferencePriority*/ inference_priority2,
    int /*TfLiteGpuInferencePriority*/ inference_priority3,
    int experimental_flags,
    int max_delegated_partitions,
    Pointer<Utf8> serialization_dir,
    Pointer<Utf8> model_token,
  ) {
    final Pointer<TfLiteGpuDelegateOptionsV2> result = calloc<TfLiteGpuDelegateOptionsV2>();
    result.ref
      ..is_precision_loss_allowed = is_precision_loss_allowed
      ..inference_preference = inference_preference
      ..inference_priority1 = inference_priority1
      ..inference_priority2 = inference_priority2
      ..inference_priority3 = inference_priority3
      ..experimental_flags = experimental_flags
      ..max_delegated_partitions = max_delegated_partitions
      ..serialization_dir = serialization_dir
      ..model_token = model_token;
    return result;
  }
}

/// Populates TfLiteGpuDelegateOptionsV2 as follows:
///   is_precision_loss_allowed = false
///   inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
///   priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION
///   priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO
///   priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO
///   experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT
///   max_delegated_partitions = 1
final TfLiteGpuDelegateOptionsV2 Function() TfLiteGpuDelegateOptionsV2Default =
    tfliteGpuLib.lookup<NativeFunction<TfLiteGpuDelegateOptionsV2 Function()>>('TfLiteGpuDelegateOptionsV2Default').asFunction();

/// Creates a new delegate instance that need to be destroyed with
/// TfLiteGpuDelegateV2Delete when delegate is no longer used by TFLite.
///
/// This delegate encapsulates multiple GPU-acceleration APIs under the hood to
/// make use of the fastest available on a device.
///
/// When `options` is set to `nullptr`, then default options are used.
final Pointer<TfLiteDelegate> Function(Pointer<TfLiteGpuDelegateOptionsV2> options) TfLiteGpuDelegateV2Create =
    tfliteGpuLib.lookup<NativeFunction<Pointer<TfLiteDelegate> Function(Pointer<TfLiteGpuDelegateOptionsV2> options)>>('TfLiteGpuDelegateV2Create').asFunction();

/// Destroys a delegate created with `TfLiteGpuDelegateV2Create` call.
final void Function(Pointer<TfLiteDelegate> delegate) TfLiteGpuDelegateV2Delete =
    tfliteGpuLib.lookup<NativeFunction<Void Function(Pointer<TfLiteDelegate> delegate)>>('TfLiteGpuDelegateV2Delete').asFunction();
