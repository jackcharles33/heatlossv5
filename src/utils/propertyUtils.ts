import { PropertyAge, PropertyType } from '../types/HouseData';
import { windowTypes } from '../constants/buildingData';

export function getWindowUValue(windowType: string): number {
  return windowTypes[windowType as keyof typeof windowTypes]?.uValue || windowTypes.double.uValue;
}

export function getAgeValue(age: PropertyAge): number {
  const values: Record<PropertyAge, number> = {
    'PRE_1960': 0,
    'BETWEEN_1960_2000': 0.33,
    'BETWEEN_2000_2008': 0.66,
    'POST_2008': 1
  };
  return values[age];
}

export function getPropertyTypeValue(type: PropertyType): number {
  const values: Record<PropertyType, number> = {
    'Detached': 1,
    'Semi-Detached / End-Terrace': 0.66,
    'End of Terrace': 0.33,
    'Terrace': 0,
    'Bungalow': 0.5
  };
  return values[type];
}