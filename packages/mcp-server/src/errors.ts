export class AppError extends Error {
  readonly code: string;
  readonly details?: Record<string, unknown>;
  readonly retryable: boolean;
  readonly httpStatus?: number;

  constructor(
    code: string,
    message: string,
    options: {
      details?: Record<string, unknown>;
      retryable?: boolean;
      httpStatus?: number;
      cause?: unknown;
    } = {}
  ) {
    super(message, { cause: options.cause });
    this.name = "AppError";
    this.code = code;
    this.details = options.details;
    this.retryable = options.retryable ?? false;
    this.httpStatus = options.httpStatus;
  }
}

export function asAppError(error: unknown): AppError {
  if (error instanceof AppError) {
    return error;
  }
  if (error instanceof Error) {
    return new AppError("internal_error", error.message, { cause: error });
  }
  return new AppError("internal_error", String(error));
}

export function errorBody(error: unknown): Record<string, unknown> {
  const appError = asAppError(error);
  return {
    ok: false,
    error: {
      code: appError.code,
      message: appError.message,
      retryable: appError.retryable,
      ...(appError.details ? { details: appError.details } : {})
    }
  };
}
