import { useState } from 'react';
import { Box, TextField, Button } from '@mui/material';

interface KrakenValidationProps {
  onValidate: (value: number) => void;
}

export function KrakenValidation({ onValidate }: KrakenValidationProps) {
  const [showInput, setShowInput] = useState(false);
  const [value, setValue] = useState('');

  const handleSubmit = () => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue) && numValue > 0) {
      onValidate(numValue);
      setShowInput(false);
      setValue('');
    }
  };

  if (!showInput) {
    return (
      <Button
        onClick={() => setShowInput(true)}
        variant="outlined"
        sx={{
          color: 'white',
          borderColor: 'rgba(255, 255, 255, 0.3)',
          padding: '16px 40px',
          fontSize: '1.1rem',
          borderRadius: '12px',
          textTransform: 'none',
          fontFamily: 'Montserrat, sans-serif',
          fontWeight: 500,
          minWidth: '240px',
          '&:hover': {
            borderColor: 'rgba(255, 255, 255, 0.5)',
            backgroundColor: 'rgba(255, 255, 255, 0.05)'
          }
        }}
      >
        Value on Kraken Field?
      </Button>
    );
  }

  return (
    <Box sx={{ 
      display: 'flex', 
      gap: 3,
      justifyContent: 'center',
      width: '100%',
      maxWidth: '480px',
      padding: '0 20px'
    }}>
      <TextField
        value={value}
        onChange={(e) => setValue(e.target.value)}
        type="number"
        placeholder="Enter Kraken value"
        size="small"
        sx={{
          width: '220px',
          '& .MuiOutlinedInput-root': {
            color: 'white',
            borderRadius: '12px',
            height: '52px',
            '& fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.3)'
            },
            '&:hover fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.5)'
            }
          }
        }}
      />
      <Button
        onClick={handleSubmit}
        variant="contained"
        sx={{
          backgroundColor: '#ff6b95',
          padding: '14px 36px',
          height: '52px',
          borderRadius: '12px',
          textTransform: 'none',
          fontFamily: 'Montserrat, sans-serif',
          fontWeight: 500,
          fontSize: '1.1rem',
          '&:hover': {
            backgroundColor: '#ff4f82'
          }
        }}
      >
        Submit
      </Button>
    </Box>
  );
}