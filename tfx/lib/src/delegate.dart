import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:quiver/check.dart';
import 'package:tfx/src/bindings/c/c_api.dart';
import 'package:tfx/src/bindings/delegates/coreml/coreml_delegate.dart';
import 'package:tfx/src/bindings/delegates/gpu/delegate.dart';
import 'package:tfx/src/bindings/delegates/gpu/metal_delegate.dart';
import 'package:tfx/src/bindings/delegates/xnnpack/xnnpack_delegate.dart';

abstract class Delegate {
  Pointer<TfLiteDelegate> get _ref;

  void delete();
}

extension DelegateExtension on Delegate {
  Pointer<TfLiteDelegate> get ref => _ref;
}

/// iOS
class CoreMlDelegate implements Delegate {
  factory CoreMlDelegate() {
    assert(Platform.isIOS);
    return CoreMlDelegate._create();
  }

  factory CoreMlDelegate.withOptions({
    int enabledDevices = TfLiteCoreMlDelegateEnabledDevices.TfLiteCoreMlDelegateDevicesWithNeuralEngine,
    int coremlVersion = 2,
    int maxDelegatedPartitions = 0,
    int minNodesPerPartition = 2,
  }) {
    assert(Platform.isIOS);
    assert(coremlVersion == 2 || coremlVersion == 3);
    final Pointer<TfLiteCoreMlDelegateOptions> options = TfLiteCoreMlDelegateOptions.allocate(
      enabledDevices,
      coremlVersion,
      maxDelegatedPartitions,
      minNodesPerPartition,
    );
    return CoreMlDelegate._create(options: options);
  }

  factory CoreMlDelegate._create({Pointer<TfLiteCoreMlDelegateOptions>? options}) {
    final Pointer<TfLiteDelegate> delegate = TfLiteCoreMlDelegateCreate(options ?? nullptr.cast());
    checkArgument(delegate.address != nullptr.address, message: 'Unable to create CoreMlDelegate.');
    return CoreMlDelegate._(delegate);
  }

  CoreMlDelegate._(Pointer<TfLiteDelegate> ref) : _ref = ref;

  @override
  final Pointer<TfLiteDelegate> _ref;
  bool _deleted = false;

  @override
  void delete() {
    checkState(!_deleted, message: 'CoreMlDelegate already deleted.');
    TfLiteCoreMlDelegateDelete(_ref);
    _deleted = true;
  }
}

/// Android
class GpuDelegateV2 implements Delegate {
  factory GpuDelegateV2() {
    assert(Platform.isAndroid);
    return GpuDelegateV2._create();
  }

  factory GpuDelegateV2.withOptions({
    int isPrecisionLossAllowed = 0,
    int inferencePreference = TfLiteGpuInferenceUsage.TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
    int inferencePriority1 = TfLiteGpuInferencePriority.TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION,
    int inferencePriority2 = TfLiteGpuInferencePriority.TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    int inferencePriority3 = TfLiteGpuInferencePriority.TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    List<int> experimentalFlags = const <int>[
      TfLiteGpuExperimentalFlags.TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT,
    ],
    int maxDelegatedPartitions = 1,
    String? serializationDir,
    String? modelToken,
  }) {
    assert(Platform.isAndroid);
    assert(isPrecisionLossAllowed == 0 || isPrecisionLossAllowed == 1);
    final Pointer<TfLiteGpuDelegateOptionsV2> options = TfLiteGpuDelegateOptionsV2.allocate(
      isPrecisionLossAllowed,
      inferencePreference,
      inferencePriority1,
      inferencePriority2,
      inferencePriority3,
      TfLiteGpuExperimentalFlags.calculateBitmask(experimentalFlags),
      maxDelegatedPartitions,
      serializationDir != null ? serializationDir.toNativeUtf8() : nullptr.cast(),
      modelToken != null ? modelToken.toNativeUtf8() : nullptr.cast(),
    );
    return GpuDelegateV2._create(options: options);
  }

  factory GpuDelegateV2._create({Pointer<TfLiteGpuDelegateOptionsV2>? options}) {
    final Pointer<TfLiteDelegate> delegate = TfLiteGpuDelegateV2Create(options ?? nullptr.cast());
    checkArgument(delegate.address != nullptr.address, message: 'Unable to create GpuDelegateV2.');
    return GpuDelegateV2._(delegate);
  }

  GpuDelegateV2._(Pointer<TfLiteDelegate> ref) : _ref = ref;

  @override
  final Pointer<TfLiteDelegate> _ref;
  bool _deleted = false;

  @override
  void delete() {
    checkState(!_deleted, message: 'GpuDelegateV2 already deleted.');
    TfLiteGpuDelegateV2Delete(_ref);
    _deleted = true;
  }
}

/// iOS
class GpuDelegate implements Delegate {
  factory GpuDelegate() {
    assert(Platform.isIOS);
    return GpuDelegate._create();
  }

  factory GpuDelegate.withOptions({
    bool allowPrecisionLoss = false,
    int waitType = TFLGpuDelegateWaitType.TFLGpuDelegateWaitTypePassive,
    bool enableQuantization = true,
  }) {
    assert(Platform.isIOS);
    final Pointer<TFLGpuDelegateOptions> options = TFLGpuDelegateOptions.allocate(
      allowPrecisionLoss,
      waitType,
      enableQuantization,
    );
    return GpuDelegate._create(options: options);
  }

  factory GpuDelegate._create({Pointer<TFLGpuDelegateOptions>? options}) {
    final Pointer<TfLiteDelegate> delegate = TFLGpuDelegateCreate(options ?? nullptr.cast());
    checkArgument(delegate.address != nullptr.address, message: 'Unable to create GpuDelegate.');
    return GpuDelegate._(delegate);
  }

  GpuDelegate._(Pointer<TfLiteDelegate> ref) : _ref = ref;

  @override
  final Pointer<TfLiteDelegate> _ref;
  bool _deleted = false;

  @override
  void delete() {
    checkState(!_deleted, message: 'GpuDelegate already deleted.');
    TFLGpuDelegateDelete(_ref);
    _deleted = true;
  }
}

/// Android/iOS
class XNNPackDelegate implements Delegate {
  factory XNNPackDelegate() {
    assert(Platform.isAndroid || Platform.isIOS);
    return XNNPackDelegate._create();
  }

  factory XNNPackDelegate.withOptions({
    int numThreads = 1,
    int flags = TFLITE_XNNPACK_DELEGATE_FLAG_QS8,
    XNNPackDelegateWeightsCache? weights_cache,
  }) {
    assert(Platform.isAndroid || Platform.isIOS);
    final Pointer<TfLiteXNNPackDelegateOptions> options = TfLiteXNNPackDelegateOptions.allocate(
      numThreads,
      flags,
      weights_cache != null ? weights_cache.ref : nullptr.cast(),
    );
    return XNNPackDelegate._create(options: options);
  }

  factory XNNPackDelegate._create({Pointer<TfLiteXNNPackDelegateOptions>? options}) {
    final Pointer<TfLiteDelegate> delegate = TfLiteXNNPackDelegateCreate(options ?? nullptr.cast());
    checkArgument(delegate.address != nullptr.address, message: 'Unable to create XNNPackDelegate.');
    return XNNPackDelegate._(delegate);
  }

  XNNPackDelegate._(Pointer<TfLiteDelegate> ref) : _ref = ref;

  @override
  final Pointer<TfLiteDelegate> _ref;
  bool _deleted = false;

  @override
  void delete() {
    checkState(!_deleted, message: 'XNNPackDelegate already deleted.');
    TfLiteXNNPackDelegateDelete(_ref);
    _deleted = true;
  }
}

class XNNPackDelegateWeightsCache {
  factory XNNPackDelegateWeightsCache() {
    assert(Platform.isAndroid || Platform.isIOS);
    final Pointer<TfLiteXNNPackDelegateWeightsCache> cache = TfLiteXNNPackDelegateWeightsCacheCreate();
    checkArgument(cache.address != nullptr.address, message: 'Unable to create XNNPackDelegateWeightsCache.');
    return XNNPackDelegateWeightsCache._(cache);
  }

  XNNPackDelegateWeightsCache._(Pointer<TfLiteXNNPackDelegateWeightsCache> ref) : _ref = ref;

  final Pointer<TfLiteXNNPackDelegateWeightsCache> _ref;
  bool _deleted = false;

  void delete() {
    checkState(!_deleted, message: 'XNNPackDelegateWeightsCache already deleted.');
    TfLiteXNNPackWeightsCacheDelete(_ref);
    _deleted = true;
  }
}

extension XNNPackDelegateWeightsCacheExtension on XNNPackDelegateWeightsCache {
  Pointer<TfLiteXNNPackDelegateWeightsCache> get ref => _ref;
}
