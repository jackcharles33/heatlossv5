import { BaseHouseData } from '../types/HouseData';

export interface StoredPrediction extends BaseHouseData {
  id?: number;
  krakenValue?: number;
  accuracy?: number;
  isAccurate?: boolean;
  confidenceScore: number;
  timestamp: string;
}