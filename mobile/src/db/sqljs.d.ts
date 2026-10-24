// The pure-JS build of sql.js has no type declarations of its own; it exports the same initializer as the main entry.
declare module 'sql.js/dist/sql-asm.js' {
  import initSqlJs from 'sql.js';
  export default initSqlJs;
}
