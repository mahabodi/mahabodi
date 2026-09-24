package ai.mahabodi;

/** Raised for any Bodi error (invalid input, missing model, unknown method, ...). */
public class BodiException extends RuntimeException {
    public BodiException(String message) { super(message); }
}
