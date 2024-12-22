import { Box, Typography, LinearProgress } from '@mui/material';
import { styled } from '@mui/material/styles';

const ProgressLabel = styled(Typography)({
  color: 'rgba(255, 255, 255, 0.7)',
  marginBottom: '0.5rem'
});

interface EfficiencyRatingProps {
  prediction: number;
}

export function EfficiencyRating({ prediction }: EfficiencyRatingProps) {
  const getEfficiencyRating = (value: number) => {
    if (value < 4) return { label: 'Excellent', color: '#4CAF50' };
    if (value < 6) return { label: 'Good', color: '#8BC34A' };
    if (value < 8) return { label: 'Average', color: '#FFC107' };
    if (value < 10) return { label: 'Poor', color: '#FF9800' };
    return { label: 'Very Poor', color: '#F44336' };
  };

  const rating = getEfficiencyRating(prediction);
  const efficiency = Math.max(0, Math.min(100, (1 - prediction/12) * 100));

  return (
    <Box sx={{ mb: 4 }}>
      <ProgressLabel>Efficiency Rating: {rating.label}</ProgressLabel>
      <LinearProgress 
        variant="determinate" 
        value={efficiency}
        sx={{
          height: 10,
          borderRadius: 5,
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          '& .MuiLinearProgress-bar': {
            backgroundColor: rating.color
          }
        }}
      />
    </Box>
  );
}