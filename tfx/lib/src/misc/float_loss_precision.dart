import 'dart:typed_data';

extension FloatLossPrecision on double {
  double get lossPrecision {
    return (ByteData(4)..setFloat32(0, this, Endian.little)).getFloat32(0, Endian.little); // 精度损失
  }
}

extension ListFloatLossPrecision on List<double> {
  List<double> get lossPrecision => map((double e) => e.lossPrecision).toList();
}
