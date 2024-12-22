import { Box, Typography } from '@mui/material';
import { AlertTriangle } from 'lucide-react';

interface ErrorDisplayProps {
  prediction: number;
  threshold?: number;
}

export function ErrorDisplay({ prediction, threshold = 10.8 }: ErrorDisplayProps) {
  if (prediction <= threshold) return null;

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        backgroundColor: 'rgba(244, 67, 54, 0.15)',
        backdropFilter: 'blur(8px)',
        border: '1px solid rgba(244, 67, 54, 0.3)',
        borderRadius: '12px',
        padding: '16px 24px',
        mb: 3,
        animation: 'pulse 2s infinite'
      }}
    >
      <AlertTriangle size={24} color="#F44336" />
      <Typography
        sx={{
          fontSize: '1.25rem',
          fontWeight: 600,
          color: '#fff',
          textShadow: '0 2px 4px rgba(0,0,0,0.2)',
          letterSpacing: '0.02em',
          flex: 1
        }}
      >
        Heat loss exceeds the {threshold} kW threshold - FAIL
      </Typography>
    </Box>
  );
}