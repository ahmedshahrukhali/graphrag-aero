import { describe, expect, it } from 'vitest';
import { bboxToOverlayStyle } from '@/components/PdfPreview';

describe('bboxToOverlayStyle', () => {
  it('converts pdfplumber bbox to top-origin percent CSS', () => {
    // 612 x 792 = US Letter in PDF points.
    const style = bboxToOverlayStyle([61.2, 79.2, 306, 396], 612, 792);
    expect(style.left).toBe('10%');
    expect(style.top).toBe('10%');
    // width  = 306 - 61.2 = 244.8 / 612 = 40%
    // height = 396 - 79.2 = 316.8 / 792 = 40%
    expect(style.width).toBe('40%');
    expect(style.height).toBe('40%');
  });

  it('clamps negative width/height to zero', () => {
    const style = bboxToOverlayStyle([100, 100, 50, 50], 200, 200);
    expect(style.width).toBe('0%');
    expect(style.height).toBe('0%');
  });
});
