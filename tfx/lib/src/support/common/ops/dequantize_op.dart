import 'package:tfx/src/bindings/c/c_api_types.dart';
import 'package:tfx/src/support/common/ops/normalize_op.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/common/ops/DequantizeOp.java

/// Dequantizes a [TensorBuffer] with given [zeroPoint] and [scale].
///
/// Note: The data type of output tensor is always [TfLiteType.kTfLiteFloat32] except when the DequantizeOp is
/// created effectively as an identity Op such as setting [zeroPoint] to 0 and [scale] to
/// 1 (in this case, the output tensor is the same instance as input).
///
/// If both [zeroPoint] and [scale] are 0, the [DequantizeOp] will be bypassed,
/// which is equivalent to setting [zeroPoint] to 0 and [scale] to 1. This can be useful
/// when passing in the quantization parameters that are extracted directly from the TFLite model
/// flatbuffer. If the tensor is not quantized, both [zeroPoint] and [scale] will be read
/// as 0.
class DequantizeOp extends NormalizeOp {
  /// Quantization: f = (q - z) * s
  DequantizeOp.from(double zeroPoint, double scale) : super.from(zeroPoint, 1 / scale);
}
