interface AreaInputs {
  floorArea: string;
  stories: number;
  glazingRatio: number;
}

interface AreaCalculations {
  wallArea: number;
  glazingArea: number;
  netWallArea: number;
  floorArea: number;
  roofArea: number;
}

export function calculateAreas(inputs: AreaInputs): AreaCalculations {
  const wallHeight = inputs.stories * 2.4;
  const wallArea = Math.sqrt(parseFloat(inputs.floorArea)) * 4 * wallHeight;
  const glazingArea = wallArea * (inputs.glazingRatio / 100);
  const netWallArea = wallArea - glazingArea;
  const floorArea = parseFloat(inputs.floorArea);
  const roofArea = floorArea;

  return {
    wallArea,
    glazingArea,
    netWallArea,
    floorArea,
    roofArea
  };
}