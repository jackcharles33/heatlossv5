import { StoredPrediction } from '../db/types';

export function getAccuracyColor(accuracy: number | undefined): string {
  if (!accuracy) return 'inherit';
  return accuracy >= 90 ? '#4CAF50' : '#F44336';
}

export function getPredictionStatus(prediction: StoredPrediction): string {
  if (!prediction.krakenValue) return '-';
  const threshold = 10.8;
  const modelCaught = prediction.heatLoss > threshold;
  const actualHigh = prediction.krakenValue > threshold;
  
  if (actualHigh && modelCaught) return 'Caught';
  if (actualHigh && !modelCaught) return 'Missed';
  if (!actualHigh && !modelCaught) return 'Correct';
  return 'False Positive';
}

export function getStatusColor(status: string): string {
  switch (status) {
    case 'Caught':
    case 'Correct':
      return '#4CAF50';
    case 'Missed':
    case 'False Positive':
      return '#F44336';
    default:
      return 'inherit';
  }
}