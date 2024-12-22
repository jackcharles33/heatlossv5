import { BaseHouseData } from '../types/HouseData';
import { wallTypes, floorTypes } from '../constants';
import { getWindowUValue } from './normalization';

export function calculateUValueFactor(input: Partial<BaseHouseData>): number {
  const wallUValue = wallTypes[input.wallType as keyof typeof wallTypes]?.uValue || 0.5;
  const floorUValue = floorTypes[input.floorType as keyof typeof floorTypes]?.uValue || 0.5;
  const windowUValue = getWindowUValue(input.windowType || 'double');
  
  return (wallUValue + floorUValue + windowUValue) / 3;
}