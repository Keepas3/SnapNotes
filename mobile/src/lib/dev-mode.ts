import Constants, { ExecutionEnvironment } from 'expo-constants';

/**
 * Whether the app may talk plain `http://` to a server on your own network. True while developing, and also in Expo Go however the bundle
 * was built (`expo start --no-dev` makes `__DEV__` false there, but Expo Go is still a development tool and allows http). An installed
 * app is neither, and refuses http anyway.
 */
export function allowsLocalHttp(): boolean {
  return __DEV__ || Constants.executionEnvironment === ExecutionEnvironment.StoreClient;
}
