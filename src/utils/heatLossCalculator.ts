import { UKRegions, wallTypes, windowTypes, floorTypes, roofTypes } from '../constants/buildingData';
import { CalculatorInputs, CalculationResults } from '../types/calculator';
import { ageMultipliers } from './ageCalculations';
import { PropertyAge } from '../types/HouseData';
import { calculateAreas } from './areaCalculations';
import { calculateElementHeatLoss } from './elementHeatLoss';

export function calculateHeatLoss(inputs: CalculatorInputs): CalculationResults {
  // Get base temperature from region
  const baseTemp = UKRegions[inputs.region as keyof typeof UKRegions]?.baseTemp || -3.2;
  const tempDiff = inputs.indoorTemp - baseTemp;
  
  // Get U-values for construction elements
  const wallUValue = wallTypes[inputs.wallType as keyof typeof wallTypes]?.uValue || 1.7;
  const windowUValue = windowTypes[inputs.windowType as keyof typeof windowTypes]?.uValue || 2.8;
  const floorUValue = floorTypes[inputs.floorType as keyof typeof floorTypes]?.uValue || 0.4;
  const roofUValue = roofTypes[inputs.roofType as keyof typeof roofTypes]?.uValue || 0.34;
  
  // Calculate areas
  const { netWallArea, glazingArea, floorArea, roofArea } = calculateAreas(inputs);
  
  // Calculate heat loss for each element (W)
  const { wallLoss, windowLoss, floorLoss, roofLoss } = calculateElementHeatLoss({
    netWallArea,
    glazingArea,
    floorArea,
    roofArea,
    wallUValue,
    windowUValue,
    floorUValue,
    roofUValue,
    tempDiff
  });
  
  // Apply age multiplier
  const ageMultiplier = ageMultipliers[inputs.age as PropertyAge] || 1;
  const totalHeatLoss = ((wallLoss + windowLoss + floorLoss + roofLoss) * ageMultiplier) / 1000;
  
  return {
    totalHeatLoss,
    breakdown: {
      walls: ((wallLoss * ageMultiplier) / 1000).toFixed(1),
      windows: ((windowLoss * ageMultiplier) / 1000).toFixed(1),
      floor: ((floorLoss * ageMultiplier) / 1000).toFixed(1),
      roof: ((roofLoss * ageMultiplier) / 1000).toFixed(1)
    }
  };
}