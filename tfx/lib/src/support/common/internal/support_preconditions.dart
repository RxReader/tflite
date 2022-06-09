class SupportPreconditions {
  const SupportPreconditions._();

  static T checkNotNull<T>(T? reference, [Object? errorMessage]) {
    if (reference == null) {
      throw ArgumentError.value(reference, null, errorMessage);
    }
    return reference;
  }

  static String checkNotEmpty(String? string, [Object? errorMessage]) {
    if (string?.isEmpty ?? true) {
      throw ArgumentError.value(string, null, errorMessage);
    }
    return string!;
  }

  static void checkArgument(bool expression, [Object? errorMessage]) {
    if (!expression) {
      throw ArgumentError.value(expression, null, errorMessage);
    }
  }

  static int checkElementIndex(int index, int size, [String? desc]) {
    return RangeError.checkValidIndex(index, null, null, size, desc);
  }

  static void checkState(bool expression, [Object? errorMessage]) {
    if (!expression) {
      throw StateError(errorMessage?.toString() ?? 'failed precondition');
    }
  }
}
