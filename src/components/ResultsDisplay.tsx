import { Paper, Typography, Box, LinearProgress, Grid } from '@mui/material';
import { styled } from '@mui/material/styles';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  background: 'rgba(255, 255, 255, 0.08)',
  backdropFilter: 'blur(10px)',
  borderRadius: '16px',
  color: '#fff',
  textAlign: 'center'
}));

const ProgressLabel = styled(Typography)({
  color: 'rgba(255, 255, 255, 0.7)',
  marginBottom: '0.5rem'
});

interface ResultsDisplayProps {
  prediction: number;
}

export function ResultsDisplay({ prediction }: ResultsDisplayProps) {
  // Calculate efficiency rating based on prediction
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
    <StyledPaper>
      <Typography variant="h5" gutterBottom>
        Heat Loss Results
      </Typography>

      <Box sx={{ my: 4 }}>
        <Typography variant="h3" sx={{ color: rating.color, mb: 1 }}>
          {prediction.toFixed(1)} kW
        </Typography>
        <Typography variant="subtitle1" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
          Total Heat Loss
        </Typography>
      </Box>

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

      <Grid container spacing={2}>
        <Grid item xs={6}>
          <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
            Annual Cost Estimate
          </Typography>
          <Typography variant="h6">
            £{Math.round(prediction * 876)} - £{Math.round(prediction * 1095)}
          </Typography>
        </Grid>
        <Grid item xs={6}>
          <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
            CO₂ Emissions
          </Typography>
          <Typography variant="h6">
            {Math.round(prediction * 1.85)} kg/year
          </Typography>
        </Grid>
      </Grid>
    </StyledPaper>
  );
}