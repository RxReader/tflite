import 'package:flutter/foundation.dart';
import 'package:tfx/src/support/common/operator.dart';
import 'package:tfx/src/support/common/processor.dart';
import 'package:tfx/src/support/image/image_processor.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/common/SequentialProcessor.java

/// A processor base class that chains a serial of [Operator] and executes them.
///
/// Typically, users could use its subclasses, e.g.
/// [ImageProcessor] rather than directly use this one.
///
/// [T] The type that the Operator is handling.
class SequentialProcessor<T> implements Processor<T> {
  @protected
  SequentialProcessor(this.operatorList) {
    final Map<String, List<int>> operatorIndex = <String, List<int>>{};
    for (Operator<T> op in operatorList) {
      final String operatorName = op.runtimeType.toString();
      final List<int> index = operatorIndex.putIfAbsent(operatorName, () => <int>[]);
      index.add(operatorList.indexOf(op));
    }
    this.operatorIndex = Map<String, List<int>>.unmodifiable(operatorIndex);
  }

  /// List of operators added to this [SequentialProcessor].
  @protected
  final List<Operator<T>> operatorList;
  /// The [Map] between the operator name and the corresponding op indexes in
  /// [operatorList]. An operator may be added multiple times into this [SequentialProcessor].
  @protected
  late final Map<String, List<int>> operatorIndex;

  @override
  T process(T input) {
    for (Operator<T> op in operatorList) {
      input = op.apply(input);
    }
    return input;
  }
}
