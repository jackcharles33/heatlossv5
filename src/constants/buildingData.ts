import { UKRegions } from './regions';
import { wallTypes, windowTypes } from './construction';
import { floorTypes } from './floorTypes';

// Reduce uninsulated loft U-value by 35%
export const roofTypes = {
  'loft-0': { name: 'Loft Uninsulated', uValue: 1.495 }, // Reduced from 2.30
  'loft-50': { name: 'Loft 50mm Insulation', uValue: 0.68 },
  'loft-100': { name: 'Loft 100mm Insulation', uValue: 0.40 },
  'loft-150': { name: 'Loft 150mm Insulation', uValue: 0.29 },
  'loft-200': { name: 'Loft 200mm Insulation', uValue: 0.20 },
  'loft-250': { name: 'Loft 250mm Insulation', uValue: 0.16 },
  'loft-270': { name: 'Loft 270mm Insulation', uValue: 0.13 },
  'loft-300': { name: 'Loft 300mm Insulation', uValue: 0.11 }
} as const;

export { UKRegions, wallTypes, windowTypes, floorTypes };