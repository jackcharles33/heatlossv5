import { Box, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';
import { Thermometer, ArrowLeftRight } from 'lucide-react';

const SpectrumContainer = styled(Box)({
  position: 'relative',
  width: '100%',
  height: '12px',
  borderRadius: '6px',
  marginTop: '32px',
  marginBottom: '64px',
  padding: '0 48px'
});

const GradientBar = styled(Box)({
  width: '100%',
  height: '100%',
  background: 'linear-gradient(to right, #2596be, #F44336)',
  borderRadius: '6px',
  position: 'relative'
});

interface MarkerProps {
  position: number;
}

const Marker = styled('div')<MarkerProps>(({ position }) => ({
  position: 'absolute',
  left: `${Math.min(Math.max(position, 0), 100)}%`,
  transform: 'translateX(-50%)',
  width: '3px',
  height: '24px',
  backgroundColor: '#fff',
  borderRadius: '2px',
  top: '-6px'
}));

const RangeIndicator = styled(Box)({
  position: 'absolute',
  top: '-40px',
  left: '50%',
  transform: 'translateX(-50%)',
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  color: 'rgba(255, 255, 255, 0.9)'
});

const Value = styled(Typography)({
  position: 'absolute',
  color: '#fff',
  fontSize: '1.5rem',
  fontWeight: 600,
  transform: 'translateX(-50%)',
  bottom: '-48px',
  whiteSpace: 'nowrap'
});

interface HeatLossSpectrumProps {
  prediction: number;
  errorMargin: number;
}

export function HeatLossSpectrum({ prediction, errorMargin }: HeatLossSpectrumProps) {
  const minValue = prediction - errorMargin;
  const maxValue = prediction + errorMargin;
  
  const getPosition = (value: number) => ((value - minValue) / (maxValue - minValue)) * 100;
  const position = getPosition(prediction);

  const formatValue = (value: number) => Number(value.toFixed(1));

  return (
    <Box sx={{ width: '100%', mt: 4, mb: 6, position: 'relative' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
        <Thermometer size={20} />
        <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
          Heat Loss Analysis
        </Typography>
      </Box>

      <SpectrumContainer>
        <RangeIndicator>
          <ArrowLeftRight size={18} />
          <Typography sx={{ fontSize: '1rem', fontWeight: 500 }}>
            Error Range
          </Typography>
        </RangeIndicator>

        <GradientBar>
          <Marker position={position} />
          <Value sx={{ left: 0, transform: 'translateX(0)' }}>
            {formatValue(minValue)} kW
          </Value>
          <Value sx={{ right: 0, left: 'auto', transform: 'translateX(0)' }}>
            {formatValue(maxValue)} kW
          </Value>
        </GradientBar>
      </SpectrumContainer>
    </Box>
  );
}