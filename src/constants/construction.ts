export const wallTypes = {
  'solid-brick-102': { name: 'Solid Brick 102mm', uValue: 2.97 },
  'solid-brick-228': { name: 'Solid Brick 228mm', uValue: 2.11 },
  'solid-brick-343': { name: 'Solid Brick 343mm', uValue: 1.64 },
  'cavity-pre60-unfilled': { name: 'Cavity Wall Pre-1960 (Unfilled)', uValue: 1.37 },
  'cavity-pre60-filled': { name: 'Cavity Wall Pre-1960 (Filled)', uValue: 0.56 },
  'cavity-post60-310': { name: 'Cavity Wall Post-1960 310mm', uValue: 0.27 },
  'cavity-post60-290-310-filled': { name: 'Cavity Wall Post-1960 290-310mm (Filled)', uValue: 0.42 },
  'cavity-post60-290-310-unfilled': { name: 'Cavity Wall Post-1960 290-310mm (Unfilled)', uValue: 0.77 },
  'cavity-post60-under290-filled': { name: 'Cavity Wall Post-1960 <290mm (Filled)', uValue: 0.45 },
  'cavity-post60-under290-unfilled': { name: 'Cavity Wall Post-1960 <290mm (Unfilled)', uValue: 0.87 },
  'timber-frame': { name: 'Timber Frame', uValue: 0.43 }
} as const;

export const windowTypes = {
  'single': { name: 'Single Glazed', uValue: 4.8 },
  'double': { name: 'Double Glazed', uValue: 1.6 },
  'triple': { name: 'Triple Glazed', uValue: 0.8 }
} as const;

export const floorTypes = {
  'concrete-uninsulated': { name: 'Concrete Uninsulated', uValue: 0.70 },
  'concrete-insulated': { name: 'Concrete Insulated', uValue: 0.30 },
  'timber-suspended': { name: 'Timber Suspended Floor', uValue: 0.80 },
  'timber-insulated': { name: 'Timber Insulated Floor', uValue: 0.30 }
} as const;