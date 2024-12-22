import { Box, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { HeatLossSpectrum } from './HeatLossSpectrum';

interface ResultsChartProps {
  prediction: number;
}

export function ResultsChart({ prediction }: ResultsChartProps) {
  const chartData = [{
    name: 'Current',
    value: prediction,
    errorY: prediction * 0.1 // 10% error margin
  }];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Heat Loss Analysis
      </Typography>

      <HeatLossSpectrum prediction={prediction} errorMargin={prediction * 0.1} />
      
      <Box sx={{ height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="name" stroke="#fff" />
            <YAxis stroke="#fff" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#2a0072', 
                border: 'none', 
                borderRadius: '8px',
                boxShadow: '0 4px 20px rgba(0,0,0,0.2)'
              }}
              labelStyle={{ color: '#fff' }}
              cursor={false}
            />
            <Bar dataKey="value" fill="#ff6b95" />
          </BarChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
}