const MAX_IMAGE_ENTRIES = 200;
const MAX_BITMAP_ENTRIES = 50;

const imagePromises = new Map();
const bitmapPromises = new Map();

const touch = (map, key, value) => {
  map.delete(key);
  map.set(key, value);
};

const trimLru = (map, maxEntries) => {
  while (map.size > maxEntries) {
    const oldestKey = map.keys().next().value;
    map.delete(oldestKey);
  }
};

export const getImage = (uri) => {
  if (!uri) return Promise.reject(new Error('Missing image uri'));

  const existing = imagePromises.get(uri);
  if (existing) {
    touch(imagePromises, uri, existing);
    return existing;
  }

  const promise = new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => {
      imagePromises.delete(uri);
      reject(new Error(`Failed to load image: ${uri}`));
    };
    image.decoding = 'async';
    image.src = uri;
  });

  imagePromises.set(uri, promise);
  trimLru(imagePromises, MAX_IMAGE_ENTRIES);
  return promise;
};

const createBitmap = async (uri) => {
  if (typeof createImageBitmap !== 'function') return await getImage(uri);
  try {
    const response = await fetch(uri);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await createImageBitmap(await response.blob());
  } catch {
    const image = await getImage(uri);
    try {
      return await createImageBitmap(image);
    } catch {
      return image;
    }
  }
};

export const getBitmap = (uri) => {
  if (!uri) return Promise.reject(new Error('Missing image uri'));

  const existing = bitmapPromises.get(uri);
  if (existing) {
    touch(bitmapPromises, uri, existing);
    return existing;
  }

  const promise = createBitmap(uri).catch((error) => {
    bitmapPromises.delete(uri);
    throw error;
  });

  bitmapPromises.set(uri, promise);
  trimLru(bitmapPromises, MAX_BITMAP_ENTRIES);
  return promise;
};

export const clearImageCache = () => {
  imagePromises.clear();
  bitmapPromises.clear();
};
