import 'package:flutter/foundation.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/label/Category.java

/// Category is a util class, contains a label, its display name, a float value as score, and the
/// index of the label in the corresponding label file. Typically it's used as result of
/// classification tasks.
@immutable
class Category {
  /// Constructs a {@link Category} object.
  /// [label] the label of this category object
  /// [displayName] the display name of the label, which may be translated for different
  ///   locales. For exmaple, a label, "apple", may be translated into Spanish for display purpose,
  ///   so that the displayName is "manzana".
  /// [score] the probability score of this label category
  /// [index] the index of the label in the corresponding label file
  factory Category.create({
    int index = _DEFAULT_INDEX,
    required String label,
    String displayName = '',
    required double score,
  }) {
    return Category._(index, label, displayName, score);
  }

  const Category._(this.index, this.label, this.displayName, this.score);

  static const int _DEFAULT_INDEX = -1;
  static const double _TOLERANCE = 1.0E-6;

  /// The index value might be -1, which means it has not been set up properly and is invalid.
  final int index;
  final String label;
  final String displayName;
  final double score;

  @override
  bool operator ==(Object other) =>
      identical(this, other) || other is Category && runtimeType == other.runtimeType && index == other.index && label == other.label && displayName == other.displayName && (other.score - score).abs() < _TOLERANCE;

  @override
  int get hashCode => index.hashCode ^ label.hashCode ^ displayName.hashCode ^ score.hashCode;
}
