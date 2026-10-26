import { devConnection } from '@/lib/settings';

describe('devConnection', () => {
  it('uses the server and token a development run was started with', () => {
    expect(devConnection('10.0.2.2:8000', ' abc ', true)).toEqual({ url: 'http://10.0.2.2:8000', token: 'abc' });
  });
  it('is off in a release build, or when either value is missing', () => {
    expect(devConnection('10.0.2.2:8000', 'abc', false)).toBeNull();
    expect(devConnection(undefined, 'abc', true)).toBeNull();
    expect(devConnection('10.0.2.2:8000', '', true)).toBeNull();
  });
});

describe('devConnection and the internet', () => {
  it('will not hand the token to an address that would send it unencrypted across the internet', () => {
    expect(devConnection('http://api.example.com', 'abc', true)).toBeNull();
    expect(devConnection('https://api.example.com', 'abc', true)).toEqual({ url: 'https://api.example.com', token: 'abc' });
    expect(devConnection('http://192.168.1.20:8000', 'abc', true)).toEqual({ url: 'http://192.168.1.20:8000', token: 'abc' });
  });
});
