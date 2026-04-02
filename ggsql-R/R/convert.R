#' Convert a data.frame to Arrow IPC stream bytes
#' @param df A data.frame.
#' @return Raw vector of Arrow IPC stream bytes.
#' @noRd
df_to_ipc <- function(df) {
  # Convert factors to character — nanoarrow doesn't support dictionary IPC encoding
  factor_cols <- vapply(df, is.factor, logical(1))
  if (any(factor_cols)) {
    df[factor_cols] <- lapply(df[factor_cols], as.character)
  }
  stream <- nanoarrow::as_nanoarrow_array_stream(df)
  con <- rawConnection(raw(0), "wb")
  on.exit(close(con))
  nanoarrow::write_nanoarrow(stream, con)
  rawConnectionValue(con)
}

#' Convert Arrow IPC stream bytes to a data.frame
#' @param ipc_bytes Raw vector of Arrow IPC stream bytes.
#' @return A data.frame.
#' @noRd
ipc_to_df <- function(ipc_bytes) {
  con <- rawConnection(ipc_bytes, "rb")
  on.exit(close(con))
  stream <- nanoarrow::read_nanoarrow(con)
  as.data.frame(stream)
}
