import React from 'react';
import { Container, Box, Typography, Paper, Grid } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { PredictionForm } from './components/PredictionForm';
import { ResultsDisplay } from './components/results/ResultsDisplay';
import { DatabaseViewer } from './components/DatabaseViewer';
import { HeatLossPredictor } from './utils/predictor';
import { HouseData } from './types/HouseData';

const theme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#180048',
      paper: 'rgba(255, 255, 255, 0.1)',
    },
    primary: {
      main: '#ff6b95',
    }
  },
  typography: {
    fontFamily: 'Montserrat, sans-serif',
    h4: {
      fontWeight: 600,
      color: '#fff',
      fontSize: '2.5rem',
      textAlign: 'center',
      marginBottom: '2rem'
    },
    h6: {
      color: '#fff',
      opacity: 0.8,
      textAlign: 'center',
      marginBottom: '3rem',
      fontWeight: 400
    }
  },
  components: {
    MuiContainer: {
      styleOverrides: {
        root: {
          paddingTop: '3rem',
          paddingBottom: '3rem'
        }
      }
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(255, 255, 255, 0.08)',
          backdropFilter: 'blur(10px)',
          borderRadius: '16px',
          padding: '2rem'
        }
      }
    }
  }
});

const predictor = new HeatLossPredictor();

function App() {
  const [prediction, setPrediction] = React.useState<number | null>(null);
  const [currentInput, setCurrentInput] = React.useState<Partial<HouseData> | null>(null);

  const handlePredict = (input: Partial<HouseData>): number => {
    try {
      const result = predictor.predict(input);
      setPrediction(result);
      setCurrentInput(input);
      return result;
    } catch (err) {
      console.error('Error making prediction:', err);
      return 0;
    }
  };

  const handleKrakenValidation = (krakenValue: number) => {
    if (prediction && currentInput) {
      predictor.validatePrediction(currentInput, prediction, krakenValue);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Typography variant="h4" component="h1">
            Heat Loss Calculator
          </Typography>
          <Typography variant="h6" component="h2">
            Predict a customer's heat loss - Qualifying Tool
          </Typography>
          
          <Grid container spacing={4} justifyContent="center">
            <Grid item xs={12} md={8}>
              <Paper elevation={3}>
                <PredictionForm onPredict={handlePredict} />
              </Paper>
            </Grid>
            
            {prediction !== null && (
              <Grid item xs={12} md={8}>
                <ResultsDisplay 
                  prediction={prediction} 
                  onKrakenValidation={handleKrakenValidation}
                />
              </Grid>
            )}

            <Grid item xs={12}>
              <DatabaseViewer />
            </Grid>
          </Grid>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;