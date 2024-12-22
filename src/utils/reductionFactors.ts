import { PropertyAge } from '../types/HouseData';

export function getReductionFactor(age: PropertyAge): number {
  switch (age) {
    case 'PRE_1960':
      return 1.1; // 10% increase
    default:
      return 0.82; // 18% reduction
  }
}