with open(r'c:\Users\vishn\Desktop\Deepfake Detection Main\extension\lib\tf.min.js', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

for api in ['setBackend', 'loadGraphModel', 'ready', 'tensor', 'zeros', 'getBackend', 'engine', 'registerBackend']:
    count = content.count(api)
    print(f'  {api}: {count} occurrences')

print(f'\nFile size: {len(content)} chars')
print(f'Has "export": {"export " in content[:2000]}')
print(f'Has "import": {"import " in content[:2000]}')

# Check if this is actually a bundled file with backends
print(f'Has "webgl": {"webgl" in content.lower()}')
print(f'Has "cpu": {"cpu" in content.lower()[:50000]}')
print(f'Has "wasm": {"wasm" in content.lower()}')
