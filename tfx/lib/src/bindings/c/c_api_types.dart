import 'dart:ffi';

/// [tensorflow#c_api_types](https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/lite/c/c_api_types.h)

/// Note that new error status values may be added in future in order to
/// indicate more fine-grained internal states, therefore, applications should
/// not rely on status values being members of the enum.
class TfLiteStatus {
  const TfLiteStatus._();

  static const int kTfLiteOk = 0;

  /// Generally referring to an error in the runtime (i.e. interpreter)
  static const int kTfLiteError = 1;

  /// Generally referring to an error from a TfLiteDelegate itself.
  static const int kTfLiteDelegateError = 2;

  /// Generally referring to an error in applying a delegate due to
  /// incompatibility between runtime and delegate, e.g., this error is returned
  /// when trying to apply a TF Lite delegate onto a model graph that's already
  /// immutable.
  static const int kTfLiteApplicationError = 3;

  /// Generally referring to serialized delegate data not being found.
  /// See tflite::delegates::Serialization.
  static const int kTfLiteDelegateDataNotFound = 4;

  /// Generally referring to data-writing issues in delegate serialization.
  /// See tflite::delegates::Serialization.
  static const int kTfLiteDelegateDataWriteError = 5;

  /// Generally referring to data-reading issues in delegate serialization.
  /// See tflite::delegates::Serialization.
  static const int kTfLiteDelegateDataReadError = 6;

  /// Generally referring to issues when the TF Lite model has ops that cannot be
  /// resolved at runtime. This could happen when the specific op is not
  /// registered or built with the TF Lite framework.
  static const int kTfLiteUnresolvedOps = 7;
}

/// Types supported by tensor
class TfLiteType {
  const TfLiteType._();

  static const int kTfLiteNoType = 0;
  static const int kTfLiteFloat32 = 1;
  static const int kTfLiteInt32 = 2;
  static const int kTfLiteUInt8 = 3;
  static const int kTfLiteInt64 = 4;
  static const int kTfLiteString = 5;
  static const int kTfLiteBool = 6;
  static const int kTfLiteInt16 = 7;
  static const int kTfLiteComplex64 = 8;
  static const int kTfLiteInt8 = 9;
  static const int kTfLiteFloat16 = 10;
  static const int kTfLiteFloat64 = 11;
  static const int kTfLiteComplex128 = 12;
  static const int kTfLiteUInt64 = 13;
  static const int kTfLiteResource = 14;
  static const int kTfLiteVariant = 15;
  static const int kTfLiteUInt32 = 16;
  static const int kTfLiteUInt16 = 17;
}

/// Legacy. Will be deprecated in favor of TfLiteAffineQuantization.
/// If per-layer quantization is specified this field will still be populated in
/// addition to TfLiteAffineQuantization.
/// Parameters for asymmetric quantization. Quantized values can be converted
/// back to float using:
///     real_value = scale * (quantized_value - zero_point)
class TfLiteQuantizationParams extends Struct {
  @Float()
  external double scale;

  @Int32()
  external int zero_point;
}
