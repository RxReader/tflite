class QuantizationParams {
  const QuantizationParams(
    this.scale,
    this.zeroPoint,
  );

  final double scale;
  final int zeroPoint;
}
