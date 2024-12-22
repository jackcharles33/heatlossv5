import { useState, useEffect } from 'react';
import { Box, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Button, IconButton } from '@mui/material';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { ModelStats } from './stats/ModelStats';
import { StoredPrediction } from '../db/types';
import { getRecentPredictions, clearPredictions } from '../db/database';
import { getAccuracyColor, getStatusColor, getPredictionStatus } from '../utils/statusUtils';

export function DatabaseViewer() {
  const [predictions, setPredictions] = useState<StoredPrediction[]>([]);
  const [showPredictions, setShowPredictions] = useState(false);

  useEffect(() => {
    const loadPredictions = async () => {
      const data = await getRecentPredictions();
      setPredictions(data);
    };
    loadPredictions();
  }, []);

  const handleClearData = async () => {
    await clearPredictions();
    setPredictions([]);
  };

  return (
    <Box sx={{ mt: 4 }}>
      <ModelStats predictions={predictions} />

      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        mt: 4,
        mb: 2,
        backgroundColor: 'rgba(255, 255, 255, 0.05)',
        padding: '12px 16px',
        borderRadius: '8px'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="h6" sx={{ mb: 0 }}>Recent Predictions</Typography>
          <IconButton 
            onClick={() => setShowPredictions(!showPredictions)}
            sx={{ color: 'white' }}
          >
            {showPredictions ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </IconButton>
        </Box>
        <Button 
          variant="outlined" 
          color="error" 
          onClick={handleClearData}
          sx={{ borderRadius: '12px' }}
        >
          Clear Data
        </Button>
      </Box>

      {showPredictions && (
        <TableContainer component={Paper} sx={{ backgroundColor: 'rgba(255, 255, 255, 0.08)' }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Timestamp</TableCell>
                <TableCell>Size</TableCell>
                <TableCell>Property Type</TableCell>
                <TableCell>Age</TableCell>
                <TableCell>Heat Loss (kW)</TableCell>
                <TableCell>Kraken Value</TableCell>
                <TableCell>Accuracy</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {predictions.map((prediction, index) => (
                <TableRow key={index}>
                  <TableCell>{new Date(prediction.timestamp).toLocaleString()}</TableCell>
                  <TableCell>{prediction.size}</TableCell>
                  <TableCell>{prediction.propertyType}</TableCell>
                  <TableCell>{prediction.age}</TableCell>
                  <TableCell>{prediction.heatLoss.toFixed(3)}</TableCell>
                  <TableCell>{prediction.krakenValue?.toFixed(3) || '-'}</TableCell>
                  <TableCell sx={{ color: getAccuracyColor(prediction.accuracy) }}>
                    {prediction.accuracy ? `${prediction.accuracy.toFixed(1)}%` : '-'}
                  </TableCell>
                  <TableCell sx={{ color: getStatusColor(getPredictionStatus(prediction)) }}>
                    {getPredictionStatus(prediction)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
}