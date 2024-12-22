import { Box, styled } from '@mui/material';

const HouseContainer = styled(Box)({
  position: 'relative',
  width: '100%',
  height: '100%',
  background: 'linear-gradient(180deg, #87CEEB 0%, #87CEEB 70%, #90EE90 70%, #90EE90 100%)'
});

const HouseImage = styled('img')({
  position: 'absolute',
  left: '50%',
  top: '50%',
  transform: 'translate(-50%, -50%)',
  maxWidth: '80%',
  maxHeight: '80%',
  objectFit: 'contain',
  pointerEvents: 'none'
});

export function HouseDiagram() {
  return (
    <HouseContainer>
      <HouseImage
        src="https://octoenergy-production-media.s3.amazonaws.com/images/Solar_house.width-800.png"
        alt="House diagram"
      />
    </HouseContainer>
  );
}