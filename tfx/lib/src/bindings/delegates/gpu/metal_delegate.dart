import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:tfx/src/bindings/c/c_api.dart';
import 'package:tfx/src/bindings/dylib.dart' show tfliteGpuLib;

/// [tensorflow#metal_delegate](https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/lite/delegates/gpu/metal_delegate.h)

class TFLGpuDelegateWaitType {
  const TFLGpuDelegateWaitType._();

  /// waitUntilCompleted
  static const int TFLGpuDelegateWaitTypePassive = 0;

  /// Minimize latency. It uses active spinning instead of mutex and consumes
  /// additional CPU resources.
  static const int TFLGpuDelegateWaitTypeActive = 1;

  /// Useful when the output is used with GPU pipeline then or if external
  /// command encoder is set.
  static const int TFLGpuDelegateWaitTypeDoNotWait = 2;

  /// Tries to avoid GPU sleep mode.
  static const int TFLGpuDelegateWaitTypeAggressive = 3;
}

/// Creates a new delegate instance that need to be destroyed with
/// DeleteFlowDelegate when delegate is no longer used by tflite.
class TFLGpuDelegateOptions extends Struct {
  /// Allows to quantify tensors, downcast values, process in float16 etc.
  @Bool()
  external bool allow_precision_loss;
  @Int32()
  external int /*TFLGpuDelegateWaitType*/ wait_type;

  /// Allows execution of integer quantized models
  @Bool()
  external bool enable_quantization;

  static Pointer<TFLGpuDelegateOptions> allocate(
    bool allow_precision_loss,
    int /*TFLGpuDelegateWaitType*/ wait_type,
    bool enable_quantization,
  ) {
    final Pointer<TFLGpuDelegateOptions> result = calloc<TFLGpuDelegateOptions>();
    result.ref
      ..allow_precision_loss = allow_precision_loss
      ..wait_type = wait_type
      ..enable_quantization = enable_quantization;
    return result;
  }
}

/// Populates TFLGpuDelegateOptions as follows:
///   allow_precision_loss = false;
///   wait_type = TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypePassive;
///   enable_quantization = true;
final TFLGpuDelegateOptions Function() TFLGpuDelegateOptionsDefault = tfliteGpuLib.lookup<NativeFunction<TFLGpuDelegateOptions Function()>>('TFLGpuDelegateOptionsDefault').asFunction();

/// Creates a new delegate instance that need to be destroyed with
/// `TFLDeleteTfLiteGpuDelegate` when delegate is no longer used by TFLite.
/// When `options` is set to `nullptr`, the following default values are used:
/// .precision_loss_allowed = false,
/// .wait_type = kPassive,
final Pointer<TfLiteDelegate> Function(Pointer<TFLGpuDelegateOptions>? options) TFLGpuDelegateCreate =
    tfliteGpuLib.lookup<NativeFunction<Pointer<TfLiteDelegate> Function(Pointer<TFLGpuDelegateOptions>? options)>>('TFLGpuDelegateCreate').asFunction();

/// Destroys a delegate created with `TFLGpuDelegateCreate` call.
final void Function(Pointer<TfLiteDelegate>) TFLGpuDelegateDelete = tfliteGpuLib.lookup<NativeFunction<Void Function(Pointer<TfLiteDelegate> delegate)>>('TFLGpuDelegateDelete').asFunction();
