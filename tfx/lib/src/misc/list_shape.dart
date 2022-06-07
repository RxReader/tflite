extension ListShape<E> on List<E> {
  List<int> get shape {
    if (isEmpty) {
      return <int>[];
    }
    dynamic list = this;
    final List<int> shape = <int>[];
    while (list is List) {
      shape.add(list.length);
      list = list.elementAt(0);
    }
    return shape;
  }

  int get computeNumElements {
    int n = 1;
    final List<int> shape = this.shape;
    for (int i = 0; i < shape.length; i++) {
      n *= shape[i];
    }
    return n;
  }

  List<T> flatten<T>() {
    final List<T> flat = <T>[];
    forEach((E element) {
      if (element is List) {
        flat.addAll(element.flatten());
      } else if (element is T) {
        flat.add(element);
      } else {
        // Error with typing
      }
    });
    return flat;
  }

  List<dynamic> reshape<T>(List<int> shape) {
    final int dimsSize = shape.length;
    int numElements = 1;
    for (int i = 0; i < dimsSize; i++) {
      numElements *= shape[i];
    }

    if (numElements != computeNumElements) {
      throw ArgumentError('Total elements mismatch expected: $numElements elements for shape: $shape but found $computeNumElements');
    }

    if (dimsSize <= 5) {
      switch (dimsSize) {
        case 2:
          return _reshape2<T>(shape);
        case 3:
          return _reshape3<T>(shape);
        case 4:
          return _reshape4<T>(shape);
        case 5:
          return _reshape5<T>(shape);
      }
    }

    List<dynamic> reshapedList = flatten<dynamic>();
    for (int i = dimsSize - 1; i > 0; i--) {
      final List<dynamic> temp = <dynamic>[];
      for (int start = 0; start + shape[i] <= reshapedList.length; start += shape[i]) {
        temp.add(reshapedList.sublist(start, start + shape[i]));
      }
      reshapedList = temp;
    }
    return reshapedList;
  }

  List<List<T>> _reshape2<T>(List<int> shape) {
    final List<T> flatList = flatten<T>();
    return List<List<T>>.generate(
      shape[0],
      (int i) => List<T>.generate(
        shape[1],
        (int j) => flatList[i * shape[1] + j],
      ),
    );
  }

  List<List<List<T>>> _reshape3<T>(List<int> shape) {
    final List<T> flatList = flatten<T>();
    return List<List<List<T>>>.generate(
      shape[0],
      (int i) => List<List<T>>.generate(
        shape[1],
        (int j) => List<T>.generate(
          shape[2],
          (int k) => flatList[i * shape[1] * shape[2] + j * shape[2] + k],
        ),
      ),
    );
  }

  List<List<List<List<T>>>> _reshape4<T>(List<int> shape) {
    final List<T> flatList = flatten<T>();
    return List<List<List<List<T>>>>.generate(
      shape[0],
      (int i) => List<List<List<T>>>.generate(
        shape[1],
        (int j) => List<List<T>>.generate(
          shape[2],
          (int k) => List<T>.generate(
            shape[3],
            (int l) => flatList[i * shape[1] * shape[2] * shape[3] + j * shape[2] * shape[3] + k * shape[3] + l],
          ),
        ),
      ),
    );
  }

  List<List<List<List<List<T>>>>> _reshape5<T>(List<int> shape) {
    final List<T> flatList = flatten<T>();
    return List<List<List<List<List<T>>>>>.generate(
      shape[0],
      (int i) => List<List<List<List<T>>>>.generate(
        shape[1],
        (int j) => List<List<List<T>>>.generate(
          shape[2],
          (int k) => List<List<T>>.generate(
            shape[3],
            (int l) => List<T>.generate(
              shape[4],
              (int m) => flatList[i * shape[1] * shape[2] * shape[3] * shape[4] + j * shape[2] * shape[3] * shape[4] + k * shape[3] * shape[4] + l * shape[4] + m],
            ),
          ),
        ),
      ),
    );
  }
}
