import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:tfx/src/bindings/c/c_api.dart';
import 'package:tfx/src/bindings/dylib.dart' show tfliteLib;

/// [tensorflow#coreml_delegate](https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/lite/delegates/coreml/coreml_delegate.h)

class TfLiteCoreMlDelegateEnabledDevices {
  const TfLiteCoreMlDelegateEnabledDevices._();

  /// Create Core ML delegate only on devices with Apple Neural Engine.
  /// Returns nullptr otherwise.
  static const int TfLiteCoreMlDelegateDevicesWithNeuralEngine = 0;

  /// Always create Core ML delegate
  static const int TfLiteCoreMlDelegateAllDevices = 1;
}

class TfLiteCoreMlDelegateOptions extends Struct {
  /// Only create delegate when Neural Engine is available on the device.
  @Int32()
  external int /*TfLiteCoreMlDelegateEnabledDevices*/ enabled_devices;

  /// Specifies target Core ML version for model conversion.
  /// Core ML 3 come with a lot more ops, but some ops (e.g. reshape) is not
  /// delegated due to input rank constraint.
  /// if not set to one of the valid versions, the delegate will use highest
  /// version possible in the platform.
  /// Valid versions: (2, 3)
  @Int32()
  external int coreml_version;

  /// This sets the maximum number of Core ML delegates created.
  /// Each graph corresponds to one delegated node subset in the
  /// TFLite model. Set this to 0 to delegate all possible partitions.
  @Int32()
  external int max_delegated_partitions;

  /// This sets the minimum number of nodes per partition delegated with
  /// Core ML delegate. Defaults to 2.
  @Int32()
  external int min_nodes_per_partition;

  static Pointer<TfLiteCoreMlDelegateOptions> allocate(
    int /*TfLiteCoreMlDelegateEnabledDevices*/ enabled_devices,
    int coreml_version,
    int max_delegated_partitions,
    int min_nodes_per_partition,
  ) {
    final Pointer<TfLiteCoreMlDelegateOptions> result = calloc<TfLiteCoreMlDelegateOptions>();
    result.ref
      ..enabled_devices = enabled_devices
      ..coreml_version = coreml_version
      ..max_delegated_partitions = max_delegated_partitions
      ..min_nodes_per_partition = min_nodes_per_partition;
    return result;
  }
}

final Pointer<TfLiteDelegate> Function(Pointer<TfLiteCoreMlDelegateOptions> options) TfLiteCoreMlDelegateCreate =
    tfliteLib.lookup<NativeFunction<Pointer<TfLiteDelegate> Function(Pointer<TfLiteCoreMlDelegateOptions> options)>>('TfLiteCoreMlDelegateCreate').asFunction();

final void Function(Pointer<TfLiteDelegate> delegate) TfLiteCoreMlDelegateDelete =
    tfliteLib.lookup<NativeFunction<Void Function(Pointer<TfLiteDelegate> delegate)>>('TfLiteCoreMlDelegateDelete').asFunction();
