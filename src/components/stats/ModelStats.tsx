import { Paper, Typography, Box } from '@mui/material';
import { StoredPrediction } from '../../db/types';

interface ModelStatsProps {
  predictions: StoredPrediction[];
}

export function ModelStats({ predictions }: ModelStatsProps) {
  const calculateStats = () => {
    const validPredictions = predictions
      .slice(1)
      .filter(p => 
        p.krakenValue !== undefined && 
        p.accuracy !== undefined && 
        !isNaN(p.accuracy)
      );

    const averageAccuracy = validPredictions.length > 0
      ? validPredictions.reduce((sum, p) => sum + (p.accuracy || 0), 0) / validPredictions.length
      : 0;

    const predictionsWithKraken = predictions.slice(1).filter(p => p.krakenValue !== undefined);
    const highHeatLossCases = predictionsWithKraken.filter(p => (p.krakenValue || 0) > 10.8).length;
    const caughtCases = predictionsWithKraken.filter(
      p => p.heatLoss > 10.8 && (p.krakenValue || 0) > 10.8
    ).length;
    const missedCases = predictionsWithKraken.filter(
      p => p.heatLoss <= 10.8 && (p.krakenValue || 0) > 10.8
    ).length;

    const caughtPercentage = highHeatLossCases > 0 ? (caughtCases / highHeatLossCases) * 100 : 0;
    const missedPercentage = highHeatLossCases > 0 ? (missedCases / highHeatLossCases) * 100 : 0;

    return {
      averageAccuracy,
      caughtPercentage,
      missedPercentage,
      totalPredictions: validPredictions.length
    };
  };

  const stats = calculateStats();

  return (
    <Paper sx={{ 
      p: 3, 
      mb: 4, 
      backgroundColor: 'rgba(255, 255, 255, 0.08)',
      backdropFilter: 'blur(10px)'
    }}>
      <Typography variant="h6" gutterBottom>Model Performance</Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 3 }}>
        <Box>
          <Typography variant="subtitle2" color="rgba(255, 255, 255, 0.7)">
            Average Accuracy
          </Typography>
          <Typography variant="h4" sx={{ color: '#4CAF50' }}>
            {stats.averageAccuracy.toFixed(1)}%
          </Typography>
          <Typography variant="caption" color="rgba(255, 255, 255, 0.5)">
            Based on {stats.totalPredictions} validated predictions
          </Typography>
        </Box>
        <Box>
          <Typography variant="subtitle2" color="rgba(255, 255, 255, 0.7)">
            High Heat Loss Detection
          </Typography>
          <Typography variant="h4" sx={{ color: '#4CAF50' }}>
            {stats.caughtPercentage.toFixed(1)}%
          </Typography>
          <Typography variant="caption" color="rgba(255, 255, 255, 0.5)">
            Caught
          </Typography>
        </Box>
        <Box>
          <Typography variant="subtitle2" color="rgba(255, 255, 255, 0.7)">
            False Negatives
          </Typography>
          <Typography variant="h4" sx={{ color: '#F44336' }}>
            {stats.missedPercentage.toFixed(1)}%
          </Typography>
          <Typography variant="caption" color="rgba(255, 255, 255, 0.5)">
            Missed
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
}