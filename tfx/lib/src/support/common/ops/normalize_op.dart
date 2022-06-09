import 'package:tfx/src/bindings/c/c_api_types.dart';
import 'package:tfx/src/support/common/internal/support_preconditions.dart';
import 'package:tfx/src/support/common/tensor_operator.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/common/ops/NormalizeOp.java

/// Normalizes a [TensorBuffer] with given mean and stddev: output = (input - mean) / stddev.
class NormalizeOp implements TensorOperator {
  /// Initializes a NormalizeOp. When being called, it creates a new [TensorBuffer], which
  /// satisfies:
  ///
  ///   output = (input - mean) / stddev
  ///
  /// In the following two cases, reset [mean] to 0 and [stddev] to 1 to bypass the
  /// normalization.
  /// 1. Both [mean] and [stddev] are 0.
  /// 2. [mean] is 0 and [stddev] is Infinity.
  ///
  /// Note: If [mean] is set to 0 and [stddev] is set to 1, no computation will
  /// happen, and original input will be directly returned in execution.
  ///
  /// Note: The returned [TensorBuffer] is always a [TfLiteType.kTfLiteFloat32] tensor at
  /// present, except when the input is a [TfLiteType.kTfLiteUInt8] tensor, [mean] is set to 0 and
  /// [stddev] is set to 1, so that the original [TfLiteType.kTfLiteUInt8] tensor is returned.
  ///
  /// [mean] the mean value to be subtracted first.
  /// [stddev] the standard deviation value to divide then.
  /// throw ArgumentError if {@code stddev} is zero.
  NormalizeOp.from(double mean, double stddev) {
    /// Make exceptions to the cases that
    /// 1. Both mean and stddev are 0.0f. This may happen when reading the normalization parameters
    /// from a tensor which does not have the values populated in the metadata. The same situation
    /// may also happen to the quantization parameters.
    /// 2. mean is 0.0f and stddev is Infinity. This may happen when reading the quantization
    /// parameters from a tensor which does not have the values populated in the metadata, and then
    /// passing the parameters into the DequantizeOp.
    /// Bypass both of the two cases, by reseting stddev to 1.0f.
    if (mean == 0.0 && (stddev == 0.0 || (stddev == double.infinity || stddev == double.negativeInfinity))) {
      stddev = 1.0;
    }

    SupportPreconditions.checkArgument(stddev != 0.0, 'Stddev cannot be zero.');
    bool meansIsZeroAndDevsIs1 = false;
    if (mean == 0.0 && stddev == 1.0) {
      meansIsZeroAndDevsIs1 = true;
    }

    _mean = <double>[mean];
    _stddev = <double>[stddev];
    _numChannels = 1;
    _isIdentityOp = meansIsZeroAndDevsIs1;
  }

  /// Initializes a NormalizeOp. When being called, it creates a new [TensorBuffer], which
  /// satisfies:
  ///
  ///   // Pseudo code. [...][i] means a certain element whose channel id is i.
  ///   output[...][i] = (input[...][i] - mean[i]) / stddev[i]
  ///
  /// Note: If all values in [mean] are set to 0 and all [stddev] are set to 1, no
  /// computation will happen, and original input will be directly returned in execution.
  ///
  /// Note: The returned [TensorBuffer] is always a [TfLiteType.kTfLiteFloat32] tensor at
  /// present, except that the input is a [TfLiteType.kTfLiteUInt8] tensor, all [mean] are set to
  /// 0 and all [stddev] are set to 1.
  ///
  /// [mean] the mean values to be subtracted first for each channel.
  /// [stddev] the standard deviation values to divide then for each channel.
  /// throw ArgumentError if any [stddev] is zero, or [mean] has different
  ///   number of elements with [stddev], or any of them is empty.
  NormalizeOp.fromList(List<double> mean, List<double> stddev) {
    SupportPreconditions.checkArgument(
      mean.length == stddev.length,
      'Per channel normalization requires same number of means and stddevs',
    );
    SupportPreconditions.checkArgument(mean.isNotEmpty, 'Means and stddevs are empty.');
    _mean = List<double>.of(mean);
    _stddev = List<double>.of(stddev);
    _numChannels = mean.length;
    bool allMeansAreZeroAndAllDevsAre1 = true;
    for (int i = 0; i < _numChannels; i++) {
      SupportPreconditions.checkArgument(stddev[i] != 0, 'Stddev cannot be zero.');
      if (stddev[i] != 1 || mean[i] != 0) {
        allMeansAreZeroAndAllDevsAre1 = false;
      }
    }
    _isIdentityOp = allMeansAreZeroAndAllDevsAre1;
  }

  /// mean.length should always be equal to stddev.length and always >= 1.
  late final List<double> _mean;
  late final List<double> _stddev;
  late final int _numChannels;
  late final bool _isIdentityOp;

  @override
  TensorBuffer apply(TensorBuffer input) {
    if (_isIdentityOp) {
      return input;
    }
    final List<int> shape = input.shape;
    SupportPreconditions.checkArgument(
        _numChannels == 1 || (shape.isNotEmpty && shape[shape.length - 1] == _numChannels),
        'Number of means (stddevs) is not same with number of channels (size of last axis).');
    // TODO(136750944): Eliminate the array copy here.
    final List<double> values = input.getFloatArray();
    int j = 0;
    for (int i = 0; i < values.length; i++) {
      values[i] = (values[i] - _mean[j]) / _stddev[j];
      j = (j + 1) % _numChannels;
    }
    TensorBuffer output;
    if (input.isDynamic) {
      output = TensorBuffer.createDynamic(TfLiteType.kTfLiteFloat32);
    } else {
      output = TensorBuffer.createFixedSize(shape, TfLiteType.kTfLiteFloat32);
    }
    output.loadFloatArray(values, shape);
    return output;
  }
}
