import { styled, Box } from '@mui/material';

export const SelectWrapper = styled(Box)({
  position: 'absolute',
  width: '240px',
  zIndex: 10,
  '& .MuiFormControl-root': {
    width: '100%'
  },
  '& .MuiSelect-root': {
    background: '#ff4cd4',
    borderRadius: '12px',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
  },
  '& .MuiOutlinedInput-root': {
    color: 'rgba(0, 0, 0, 0.87)',
    fontSize: '0.95rem',
    fontFamily: 'Montserrat, sans-serif',
    backgroundColor: '#ff4cd4',
    '& .MuiSvgIcon-root': {
      color: 'rgba(0, 0, 0, 0.54)'
    }
  },
  '& .MuiSelect-select': {
    padding: '12px 16px'
  },
  '& .MuiMenuItem-root': {
    fontFamily: 'Montserrat, sans-serif'
  },
  '& .MuiTypography-root': { // Label styling
    color: '#000',
    fontWeight: 600,
    fontSize: '0.95rem',
    marginBottom: '8px',
    fontFamily: 'Montserrat, sans-serif'
  }
});