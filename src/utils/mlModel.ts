import { BaseHouseData } from '../types/HouseData';
import { wallTypes, floorTypes } from '../constants/buildingData';
import { normalizePropertyType, normalizeAge, getWindowUValue } from './normalization';
import { calculateUValueFactor } from './uValueCalculations';

export class MLModel {
  private weights: number[];
  private bias: number;
  private learningRate: number;
  private maxSize: number;

  constructor() {
    this.weights = Array(9).fill(0).map(() => Math.random() * 0.01);
    this.bias = Math.random() * 0.01;
    this.learningRate = 0.0001;
    this.maxSize = 300;
  }

  train(data: BaseHouseData[], epochs: number = 2000): void {
    this.maxSize = Math.max(...data.map(house => house.size));

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const house of data) {
        const input = this.normalizeInput(house);
        const prediction = this.predict(input);
        const error = house.heatLoss - prediction;

        for (let i = 0; i < this.weights.length; i++) {
          this.weights[i] += this.learningRate * error * input[i];
        }
        this.bias += this.learningRate * error;
      }
    }
  }

  predict(input: number[]): number {
    return input.reduce((sum, value, index) => sum + value * this.weights[index], 0) + this.bias;
  }

  private normalizeInput(input: Partial<BaseHouseData>): number[] {
    const size = input.size || 100;
    const propertyType = input.propertyType || 'Detached';
    const wallType = input.wallType || 'cavity-post60-290-310-filled';
    const floorType = input.floorType || 'concrete-75';
    const windowType = input.windowType || 'double';

    const sizeNorm = size / this.maxSize;
    const wallUValue = wallTypes[wallType as keyof typeof wallTypes]?.uValue || 0.42;
    const floorUValue = floorTypes[floorType as keyof typeof floorTypes]?.uValue || 0.28;
    const windowUValue = getWindowUValue(windowType);

    return [
      sizeNorm,
      normalizeAge(input.age),
      normalizePropertyType(propertyType),
      wallUValue / 3.0,
      floorUValue / 0.7,
      windowUValue / 4.8,
      size / this.maxSize,
      1.0
    ];
  }

  predictHeatLoss(input: Partial<BaseHouseData>): number {
    const normalizedInput = this.normalizeInput(input);
    const prediction = this.predict(normalizedInput);
    const uValueFactor = calculateUValueFactor(input);
    return Math.max(0, prediction * (1 + (uValueFactor * 0.25)));
  }
}