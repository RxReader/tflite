import 'dart:typed_data';

import 'package:tfx/src/support/common/internal/support_preconditions.dart';
import 'package:tfx/src/support/label/category.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/label/TensorLabel.java

/// TensorLabel is an util wrapper for TensorBuffers with meaningful labels on an axis.
///
/// For example, an image classification model may have an output tensor with shape as [1, 10],
/// where 1 is the batch size and 10 is the number of categories. In fact, on the 2nd axis, we could
/// label each sub-tensor with the name or description of each corresponding category.
/// [TensorLabel] could help converting the plain Tensor in [TensorBuffer] into a map from
/// predefined labels to sub-tensors. In this case, if provided 10 labels for the 2nd axis,
/// [TensorLabel] could convert the original [1, 10] Tensor to a 10 element map, each value of which
/// is Tensor in shape {} (scalar). Usage example:
///
///   TensorBuffer outputTensor = ...;
///   List<String> labels = FileUtil.loadLabels(labelFilePath);
///   // labels the first axis with size greater than one
///   TensorLabel labeled = new TensorLabel(labels, outputTensor);
///   // If each sub-tensor has effectively size 1, we can directly get a float value
///   Map<String, double> probabilities = labeled.getMapWithFloatValue();
///   // Or get sub-tensors, when each sub-tensor has elements more than 1
///   Map<String, TensorBuffer> subTensors = labeled.getMapWithTensorBuffer();
///
/// Note: currently we only support tensor-to-map conversion for the first label with size greater
/// than 1.
///
/// [FileUtil.loadLabels(String)] to load labels from a label file (plain text file whose each line is a label) in assets simply.
class TensorLabel {
  /// Creates a TensorLabel object which is able to label on one axis of multi-dimensional tensors.
  /// Note: The labels are applied on the first axis whose size is larger than 1. For example, if
  /// the shape of the tensor is [1, 10, 3], the labels will be applied on axis 1 (id starting from
  /// 0), and size of [axisLabels] should be 10 as well.
  ///
  /// [axisLabels] A list of labels, whose size should be same with the size of the tensor on
  ///   the to-be-labeled axis.
  /// [tensorBuffer] The TensorBuffer to be labeled.
  factory TensorLabel.fromList({
    required List<String> axisLabels,
    required TensorBuffer tensorBuffer,
  }) {
    return TensorLabel.fromMap(
      axisLabels: _makeMap(_getFirstAxisWithSizeGreaterThanOne(tensorBuffer), axisLabels),
      tensorBuffer: tensorBuffer,
    );
  }

  /// Creates a TensorLabel object which is able to label on the axes of multi-dimensional tensors.
  ///
  /// [axisLabels] A map, whose key is axis id (starting from 0) and value is corresponding
  ///   labels. Note: The size of labels should be same with the size of the tensor on that axis.
  /// [tensorBuffer] The TensorBuffer to be labeled.
  /// throw ArgumentError if any key in [axisLabels] is out of range (compared to
  ///   the shape of [tensorBuffer], or any value (labels) has different size with the
  ///   [tensorBuffer] on the given dimension.
  TensorLabel.fromMap({
    required Map<int, List<String>> axisLabels,
    required TensorBuffer tensorBuffer,
  }) {
    _axisLabels = axisLabels;
    _tensorBuffer = tensorBuffer;
    _shape = tensorBuffer.shape;
    for (MapEntry<int, List<String>> entry in axisLabels.entries) {
      final int axis = entry.key;
      SupportPreconditions.checkArgument(axis >= 0 && axis < _shape.length, 'Invalid axis id: $axis');
      SupportPreconditions.checkArgument(
        _shape[axis] == entry.value.length,
        'Label number ${entry.value.length} mismatch the shape on axis $axis',
      );
    }
  }

  late final Map<int, List<String>> _axisLabels;
  late final TensorBuffer _tensorBuffer;
  late final List<int> _shape;

  /// Gets the map with a pair of the label and the corresponding TensorBuffer. Only allow the
  /// mapping on the first axis with size greater than 1 currently.
  Map<String, TensorBuffer> getMapWithTensorBuffer() {
    final int labeledAxis = _getFirstAxisWithSizeGreaterThanOne(_tensorBuffer);

    SupportPreconditions.checkArgument(
      _axisLabels.containsKey(labeledAxis),
      'get a <String, TensorBuffer> map requires the labels are set on the first non-1 axis.',
    );

    final Map<String, TensorBuffer> labelToTensorMap = <String, TensorBuffer>{};
    final List<String> labels = _axisLabels[labeledAxis]!;

    final int dataType = _tensorBuffer.dataType;
    final int typeSize = _tensorBuffer.typeSize;
    final int flatSize = _tensorBuffer.flatSize;

    // Gets the underlying bytes that could be used to generate the sub-array later.
    final ByteBuffer byteBuffer = _tensorBuffer.buffer;

    // Note: computation below is only correct when labeledAxis is the first axis with size greater
    // than 1.
    final int subArrayLength = (flatSize / _shape[labeledAxis]).floor() * typeSize;
    int i = 0;
    for (String label in labels) {
      final ByteData byteData = byteBuffer.asByteData(i * subArrayLength);
      final TensorBuffer labelBuffer = TensorBuffer.createDynamic(dataType: dataType);
      labelBuffer.loadBuffer(byteData.buffer, _shape.sublist(labeledAxis + 1, _shape.length));
      labelToTensorMap[label] = labelBuffer;
      i += 1;
    }
    return labelToTensorMap;
  }

  /// Gets a map that maps label to float. Only allow the mapping on the first axis with size greater
  /// than 1, and the axis should be effectively the last axis (which means every sub tensor
  /// specified by this axis should have a flat size of 1).
  ///
  /// [TensorLabel.getCategoryList] is an alternative API to get the result.
  Map<String, double> getMapWithFloatValue() {
    final int labeledAxis = _getFirstAxisWithSizeGreaterThanOne(_tensorBuffer);
    SupportPreconditions.checkState(
      labeledAxis == _shape.length - 1,
      'get a <String, Scalar> map is only valid when the only labeled axis is the last one.',);
    final List<String>? labels = _axisLabels[labeledAxis];
    final List<double> data = _tensorBuffer.getFloatArray();
    SupportPreconditions.checkState(labels?.length == data.length);
    final Map<String, double> result = <String, double>{};
    int i = 0;
    for (String label in labels!) {
      result[label] = data[i];
      i += 1;
    }
    return result;
  }

  /// Gets a list of [Category] from the [TensorLabel] object.
  ///
  /// The axis of label should be effectively the last axis (which means every sub tensor
  /// specified by this axis should have a flat size of 1), so that each labelled sub tensor could be
  /// converted into a float value score. Example: A [TensorLabel] with shape [code [2, 5, 3]]
  /// and axis 2 is valid. If axis is 1 or 0, it cannot be converted into a [Category].
  ///
  /// [TensorLabel.getMapWithFloatValue] is an alternative but returns a {@link Map} as
  /// the result.
  ///
  /// throw StateError if size of a sub tensor on each label is not 1.
  List<Category> getCategoryList() {
    final int labeledAxis = _getFirstAxisWithSizeGreaterThanOne(_tensorBuffer);
    SupportPreconditions.checkState(
      labeledAxis == _shape.length - 1,
      'get a Category list is only valid when the only labeled axis is the last one.',);
    final List<String>? labels = _axisLabels[labeledAxis];
    final List<double> data = _tensorBuffer.getFloatArray();
    SupportPreconditions.checkState(labels?.length == data.length);
    final List<Category> result = <Category>[];
    int i = 0;
    for (String label in labels!) {
      result.add(Category.create(label: label, score: data[i]));
      i += 1;
    }
    return result;
  }

  static int _getFirstAxisWithSizeGreaterThanOne(TensorBuffer tensorBuffer) {
    final List<int> shape = tensorBuffer.shape;
    for (int i = 0; i < shape.length; i++) {
      if (shape[i] > 1) {
        return i;
      }
    }
    throw ArgumentError('Cannot find an axis to label. A valid axis to label should have size larger than 1.');
  }

  // Helper function to wrap the List<String> to a one-entry map.
  static Map<int, List<String>> _makeMap(int axis, List<String> labels) {
    return <int, List<String>>{
      axis: labels,
    };
  }
}
