/**
 * NIfTI-1 I/O — nnU-Net / SimpleITK axis convention.
 *
 * SimpleITK reads a NIfTI and returns an array of shape (nz, ny, nx):
 * the z-axis (slice direction) is slowest-varying, x is fastest-varying.
 * NIfTI-1 also stores data with x fastest in memory, so the flat byte layout
 * is identical:  flat_index(z,y,x) = z*ny*nx + y*nx + x  ==  x + y*nx + z*nx*ny.
 * No data transposition is needed — we only reinterpret the dimension labels.
 *
 * parseNifti  returns  dims=[nz,ny,nx]  and  pixdim=[sz,sy,sx]
 * encodeNifti expects  those same conventions  and writes a valid NIfTI-1 file
 */

export async function decompressIfGzip(arrayBuffer) {
  const magic = new Uint8Array(arrayBuffer, 0, 2);
  if (magic[0] !== 0x1f || magic[1] !== 0x8b) return arrayBuffer;
  const ds = new DecompressionStream('gzip');
  const writer = ds.writable.getWriter();
  writer.write(new Uint8Array(arrayBuffer));
  writer.close();
  return new Response(ds.readable).arrayBuffer();
}

/**
 * Parse a NIfTI-1 file (already decompressed).
 *
 * Returns:
 *   data      Float32Array  voxel intensities (scl_slope/inter applied)
 *   dims      [nz, ny, nx]  spatial sizes — nnU-Net / SimpleITK order
 *   pixdim    [sz, sy, sx]  voxel spacings — nnU-Net / SimpleITK order
 *   origDims  [nx, ny, nz]  NIfTI native header order (needed by encodeNifti)
 *   header    ArrayBuffer   first voxOffset bytes of original file
 *   datatype, sclSlope, sclInter  (diagnostics)
 */
export function parseNifti(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const le   = view.getInt32(0, true) === 348 || view.getInt16(0, true) === 348;

  // NIfTI dim[] header: dim[1]=nx (x), dim[2]=ny (y), dim[3]=nz (z)
  const nx = view.getInt16(42, le);
  const ny = view.getInt16(44, le);
  const nz = view.getInt16(46, le);

  // NIfTI pixdim[]: pixdim[1]=sx, pixdim[2]=sy, pixdim[3]=sz
  const sx = Math.abs(view.getFloat32(80, le));
  const sy = Math.abs(view.getFloat32(84, le));
  const sz = Math.abs(view.getFloat32(88, le));

  const datatype  = view.getInt16(70, le);
  const sclSlope  = view.getFloat32(112, le);
  const sclInter  = view.getFloat32(116, le);
  const voxOffset = Math.max(352, view.getFloat32(108, le));
  const raw = new Uint8Array(arrayBuffer, voxOffset);
  const n   = nx * ny * nz;

  let data;
  switch (datatype) {
    case 2:  data = new Float32Array(new Uint8Array (raw.buffer, raw.byteOffset, n)); break;
    case 4:  data = new Float32Array(new Int16Array (raw.buffer, raw.byteOffset, n)); break;
    case 8:  data = new Float32Array(new Int32Array (raw.buffer, raw.byteOffset, n)); break;
    case 16: data = new Float32Array(raw.buffer, raw.byteOffset, n); break;
    case 64: {
      const f64 = new Float64Array(raw.buffer, raw.byteOffset, n);
      data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = f64[i];
      break;
    }
    default: throw new Error(`Unsupported NIfTI datatype: ${datatype}`);
  }

  // Apply NIfTI intensity scaling (slope == 0 means "no scaling applied")
  if (sclSlope !== 0 && (sclSlope !== 1 || sclInter !== 0)) {
    for (let i = 0; i < n; i++) data[i] = data[i] * sclSlope + sclInter;
  }

  const header = arrayBuffer.slice(0, voxOffset);
  return {
    data,
    // nnU-Net / SimpleITK convention: z (slice axis) first
    dims:     [nz, ny, nx],
    pixdim:   [sz, sy, sx],
    // Original NIfTI header order: needed to write the output mask correctly
    origDims: [nx, ny, nz],
    header,
    datatype,
    sclSlope,
    sclInter,
  };
}

/**
 * Encode a binary mask as NIfTI-1, reusing the original image header.
 *
 * maskData  Uint8Array  in nnU-Net (nz, ny, nx) order
 * origDims  [nx, ny, nz]  from parseNifti().origDims
 * origHeader  ArrayBuffer  from parseNifti().header
 *
 * The flat memory layout of maskData is already correct for NIfTI:
 *   z*ny*nx + y*nx + x  ==  x + y*nx + z*nx*ny  (same index, same bytes)
 */
export function encodeNifti(maskData, origDims, origHeader) {
  const [nx, ny, nz] = origDims;
  const n            = nx * ny * nz;
  const HEADER_SIZE  = 352;

  const out    = new ArrayBuffer(HEADER_SIZE + n);
  const hBytes = new Uint8Array(out, 0, HEADER_SIZE);
  hBytes.set(new Uint8Array(origHeader).slice(0, Math.min(origHeader.byteLength, HEADER_SIZE)));

  const view = new DataView(out);
  const le   = true;
  view.setInt32(0,   348,         le);  // sizeof_hdr
  view.setInt16(40,  3,           le);  // ndim
  view.setInt16(42,  nx,          le);  // dim[1]
  view.setInt16(44,  ny,          le);  // dim[2]
  view.setInt16(46,  nz,          le);  // dim[3]
  view.setInt16(48,  1,           le);  // dim[4] = 1 timepoint
  view.setInt16(70,  2,           le);  // datatype: UINT8
  view.setInt16(72,  8,           le);  // bitpix: 8
  view.setFloat32(108, HEADER_SIZE, le);
  view.setFloat32(112, 1.0,       le);  // scl_slope
  view.setFloat32(116, 0.0,       le);  // scl_inter
  view.setFloat32(124, 1.0,       le);  // cal_max = 1 (overrides inherited T1w value)
  view.setFloat32(128, 0.0,       le);  // cal_min = 0
  view.setInt32(140,   1,         le);  // glmax
  view.setInt32(144,   0,         le);  // glmin
  view.setUint8(344, 0x6E);             // 'n'
  view.setUint8(345, 0x2B);             // '+'
  view.setUint8(346, 0x31);             // '1'
  view.setUint8(347, 0x00);

  new Uint8Array(out, HEADER_SIZE).set(maskData.slice(0, n));
  return out;
}
