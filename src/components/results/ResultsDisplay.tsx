import { Paper, Typography, Box } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useState, useEffect } from 'react';
import { ErrorDisplay } from './ErrorDisplay';
import { HeatLossSpectrum } from './HeatLossSpectrum';
import { KrakenValidation } from './KrakenValidation';
import { getAverageAccuracy, calculateErrorMargin } from '../../utils/accuracyCalculations';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  background: 'rgba(255, 255, 255, 0.08)',
  backdropFilter: 'blur(10px)',
  borderRadius: '16px',
  color: '#fff',
  textAlign: 'center'
}));

const ResultValue = styled(Typography)({
  fontSize: '4rem',
  fontWeight: 700,
  letterSpacing: '-0.02em',
  background: 'linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.8) 100%)',
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  textShadow: '0 2px 10px rgba(255,255,255,0.2)',
  marginBottom: '0.5rem'
});

const ResultLabel = styled(Typography)({
  fontSize: '1.25rem',
  fontWeight: 500,
  color: 'rgba(255,255,255,0.9)',
  textTransform: 'none',
  letterSpacing: '0.05em'
});

interface ResultsDisplayProps {
  prediction: number;
  onKrakenValidation?: (value: number) => void;
}

export function ResultsDisplay({ prediction, onKrakenValidation }: ResultsDisplayProps) {
  const [errorMargin, setErrorMargin] = useState<number>(prediction * 0.1);

  useEffect(() => {
    const updateErrorMargin = async () => {
      const accuracy = await getAverageAccuracy();
      setErrorMargin(calculateErrorMargin(prediction, accuracy));
    };
    updateErrorMargin();
  }, [prediction]);

  const formatNumber = (num: number) => {
    return Number(num.toPrecision(3)).toString();
  };

  return (
    <StyledPaper elevation={4}>
      <Typography 
        variant="h5" 
        sx={{ 
          fontWeight: 600,
          fontSize: '1.5rem',
          mb: 3,
          color: 'rgba(255,255,255,0.95)'
        }}
      >
        Heat Loss Results
      </Typography>

      <ErrorDisplay prediction={prediction} />

      <Box sx={{ my: 5 }}>
        <ResultValue>
          {formatNumber(prediction)} kW
        </ResultValue>
        <ResultLabel>
          Heat Loss Estimate
        </ResultLabel>
      </Box>
      
      <Box sx={{ my: 6 }}>
        <HeatLossSpectrum prediction={prediction} errorMargin={errorMargin} />
      </Box>

      {onKrakenValidation && (
        <Box sx={{ 
          mt: 6, 
          borderTop: '1px solid rgba(255, 255, 255, 0.1)', 
          pt: 6, 
          pb: 3,
          display: 'flex',
          justifyContent: 'center'
        }}>
          <KrakenValidation onValidate={onKrakenValidation} />
        </Box>
      )}
    </StyledPaper>
  );
}