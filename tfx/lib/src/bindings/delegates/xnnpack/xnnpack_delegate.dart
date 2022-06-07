import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:tfx/src/bindings/c/c_api.dart';
import 'package:tfx/src/bindings/dylib.dart' show tfliteLib;

/// [tensorflow#xnnpack_delegate](https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h)

/// Enable XNNPACK acceleration for signed quantized 8-bit inference.
/// This includes operators with channel-wise quantized weights.
const int TFLITE_XNNPACK_DELEGATE_FLAG_QS8 = 0x00000001;
/// Enable XNNPACK acceleration for unsigned quantized 8-bit inference.
const int TFLITE_XNNPACK_DELEGATE_FLAG_QU8 = 0x00000002;
/// Force FP16 inference for FP32 operators.
const int TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16 = 0x00000004;

class TfLiteXNNPackDelegateWeightsCache extends Opaque {}

class TfLiteXNNPackDelegateOptions extends Struct {
  /// Number of threads to use in the thread pool.
  /// 0 or negative value means no thread pool used.
  @Int32()
  external int num_threads;
  /// Bitfield with any combination of the following binary options:
  /// - TFLITE_XNNPACK_DELEGATE_FLAG_QS8
  /// - TFLITE_XNNPACK_DELEGATE_FLAG_QU8
  /// - TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16
  @Int32()
  external int flags;
  /// Cache for packed weights, can be shared between multiple instances of
  /// delegates.
  external Pointer<TfLiteXNNPackDelegateWeightsCache> weights_cache;

  static Pointer<TfLiteXNNPackDelegateOptions> allocate(
    int num_threads,
    int flags,
    Pointer<TfLiteXNNPackDelegateWeightsCache> weights_cache,
  ) {
    final Pointer<TfLiteXNNPackDelegateOptions> result = calloc<TfLiteXNNPackDelegateOptions>();
    result.ref
      ..num_threads = num_threads
      ..flags = flags
      ..weights_cache = weights_cache;
    return result;
  }
}

/// Returns a structure with the default XNNPack delegate options.
final TfLiteXNNPackDelegateOptions Function() TfLiteXNNPackDelegateOptionsDefault =
    tfliteLib.lookup<NativeFunction<TfLiteXNNPackDelegateOptions Function()>>('TfLiteXNNPackDelegateOptionsDefault').asFunction();

/// Creates a new delegate instance that need to be destroyed with
/// `TfLiteXNNPackDelegateDelete` when delegate is no longer used by TFLite.
/// When `options` is set to `nullptr`, default values are used (see
/// implementation of TfLiteXNNPackDelegateOptionsDefault in the .cc file for
/// details).
final Pointer<TfLiteDelegate> Function(Pointer<TfLiteXNNPackDelegateOptions> options) TfLiteXNNPackDelegateCreate =
    tfliteLib.lookup<NativeFunction<Pointer<TfLiteDelegate> Function(Pointer<TfLiteXNNPackDelegateOptions> options)>>('TfLiteXNNPackDelegateCreate').asFunction();

/// Destroys a delegate created with `TfLiteXNNPackDelegateCreate` call.
final void Function(Pointer<TfLiteDelegate> delegate) TfLiteXNNPackDelegateDelete = tfliteLib.lookup<NativeFunction<Void Function(Pointer<TfLiteDelegate> delegate)>>('TfLiteXNNPackDelegateDelete').asFunction();

/// Creates a new weights cache that can be shared with multiple delegate
/// instances.
final Pointer<TfLiteXNNPackDelegateWeightsCache> Function() TfLiteXNNPackDelegateWeightsCacheCreate =
  tfliteLib.lookup<NativeFunction<Pointer<TfLiteXNNPackDelegateWeightsCache> Function()>>('TfLiteXNNPackDelegateWeightsCacheCreate').asFunction();

/// Destroys a weights cache created with
/// `TfLiteXNNPackDelegateWeightsCacheCreate` call.
final void Function(Pointer<TfLiteXNNPackDelegateWeightsCache> cache) TfLiteXNNPackWeightsCacheDelete = tfliteLib.lookup<NativeFunction<Void Function(Pointer<TfLiteXNNPackDelegateWeightsCache> cache)>>('TfLiteXNNPackWeightsCacheDelete').asFunction();
