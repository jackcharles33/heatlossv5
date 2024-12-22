export interface CalculatorInputs {
  region: string;
  postcode: string;
  wallType: string;
  windowType: string;
  floorType: string;
  roofType: string;
  floorArea: string;
  stories: number;
  indoorTemp: number;
  glazingRatio: number;
  age: string;
}

export interface HeatLossBreakdown {
  walls: string;
  windows: string;
  floor: string;
  roof: string;
}

export interface CalculationResults {
  totalHeatLoss: number;
  breakdown: HeatLossBreakdown;
}