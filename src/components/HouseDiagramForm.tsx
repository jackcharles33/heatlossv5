import { Box, Button } from '@mui/material';
import { styled } from '@mui/material/styles';
import { PropertyTypeSelect } from './form/PropertyTypeSelect';
import { PropertyAgeSelect } from './form/PropertyAgeSelect';
import { FloorAreaInput } from './form/FloorAreaInput';
import { FloorTypeSelect } from './form/FloorTypeSelect';
import { WallTypeSelect } from './form/WallTypeSelect';
import { WindowTypeSelect } from './form/WindowTypeSelect';
import { RoofTypeSelect } from './form/RoofTypeSelect';
import { HouseDiagram } from './HouseDiagram';
import { StepHeading } from './styles/StepHeading';
import { SelectWrapper } from './styles/SelectWrapper';

const HouseDiagramContainer = styled(Box)({
  position: 'relative',
  width: '100%',
  height: '600px',
  overflow: 'hidden',
  borderRadius: '24px',
  marginBottom: '60px' // Increased margin bottom
});

interface HouseDiagramFormProps {
  values: any;
  onChange: (event: any) => void;
  onSubmit: (e: React.FormEvent) => void;
}

export function HouseDiagramForm({ values, onChange, onSubmit }: HouseDiagramFormProps) {
  return (
    <Box 
      component="form" 
      onSubmit={onSubmit}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        gap: 3,
        width: '100%',
        margin: '0 auto',
        pt: 0
      }}
    >
      <Box>
        <StepHeading>
          Step 1: Select age and property type
        </StepHeading>

        <Box sx={{ 
          display: 'flex', 
          gap: 4,
          width: '100%',
          maxWidth: '740px',
          margin: '0 auto',
          justifyContent: 'center',
          '& .MuiFormControl-root': {
            width: '220px'
          }
        }}>
          <PropertyTypeSelect value={values.propertyType} onChange={onChange} />
          <PropertyAgeSelect value={values.age} onChange={onChange} />
          <FloorAreaInput value={values.size} onChange={onChange} />
        </Box>
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <StepHeading>
          Step 2: Select insulation level
        </StepHeading>

        <HouseDiagramContainer>
          <HouseDiagram />
          
          <SelectWrapper sx={{ top: '36%', left: '53%', transform: 'translateX(-50%)' }}>
            <RoofTypeSelect value={values.roofType} onChange={onChange} />
          </SelectWrapper>

          <SelectWrapper sx={{ top: '53%', left: '2%' }}>
            <WallTypeSelect value={values.wallType} onChange={onChange} age={values.age} />
          </SelectWrapper>

          <SelectWrapper sx={{ top: '48%', right: '2%' }}>
            <WindowTypeSelect value={values.windowType} onChange={onChange} />
          </SelectWrapper>

          <SelectWrapper sx={{ bottom: '50px', left: '50%', transform: 'translateX(-50%)' }}>
            <FloorTypeSelect value={values.floorType} onChange={onChange} />
          </SelectWrapper>
        </HouseDiagramContainer>

        <Button 
          type="submit" 
          variant="contained" 
          sx={{ 
            width: '100%',
            maxWidth: '400px',
            height: '48px',
            borderRadius: '12px',
            fontSize: '1.1rem',
            margin: '0 auto',
            display: 'block',
            color: '#fff',
            backgroundColor: '#ff4cd4',
            fontFamily: 'Montserrat, sans-serif',
            textTransform: 'none',
            '&:hover': {
              backgroundColor: '#e43dbd'
            }
          }}
        >
          Calculate heat loss
        </Button>
      </Box>
    </Box>
  );
}