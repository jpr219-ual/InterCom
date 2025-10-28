#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

'''Feedback suppression with frequency domain subtraction.'''

import logging
import numpy as np
import minimal
import buffer
        
class Feedback_Suppression(buffer.Buffering):
    def __init__(self):
        super().__init__()
        logging.info(__doc__)
        
        # Parámetros del filtro LMS
        self.mu = 0.1  # Learning rate (tasa de adaptación)
        self.n_taps = max(1, 2 * minimal.args.frames_per_chunk // minimal.args.number_of_channels)  # Longitud del filtro (mínimo 1)
        # Coeficientes por canal
        self.filter_coeffs = np.zeros((int(minimal.args.number_of_channels), int(self.n_taps)), dtype=np.float32)
        
        # Buffer para el historial de salida (para estimar el feedback)
        self.output_history = np.zeros((self.chunks_to_buffer * 2, 
                                      minimal.args.frames_per_chunk, 
                                      minimal.args.number_of_channels), 
                                     dtype=np.float32)

        # Inicializaciones necesarias para evitar AttributeError en el callback
        # previous_output existe desde el principio (chunks de ceros)
        try:
            self.previous_output = self.zero_chunk.copy()
        except Exception:
            # Fallback por si acaso
            self.previous_output = np.zeros((minimal.args.frames_per_chunk, minimal.args.number_of_channels), dtype=np.int16)

        # Índice de historial y estimación inicial del feedback
        self.history_index = 0
        self.feedback_estimate = self.previous_output.astype(np.float32).copy()

    def _lms_filter(self, input_signal):
        """Implementación simplificada y robusta del filtro LMS.
        Devuelve la señal de entrada con la estimación de feedback restada.
        """
        N = input_signal.shape[0]
        C = int(minimal.args.number_of_channels)
        n_taps = int(min(self.n_taps, N))  # no puede exceder N
        eps = 1e-8

        # Reservar salida en float para evitar overflow durante cálculo
        cleaned = np.zeros((N, C), dtype=np.float32)

        for ch in range(C):
            x = input_signal[:, ch].astype(np.float32) / 32768.0  # señal de entrada normalizada
            ref = self.previous_output[:, ch].astype(np.float32) / 32768.0  # referencia (salida anterior)

            # Construir matriz de retardos X (N x n_taps)
            X = np.zeros((N, n_taps), dtype=np.float32)
            for tap in range(n_taps):
                if tap == 0:
                    X[:, tap] = x
                else:
                    X[tap:, tap] = x[:-tap]

            # Estimación actual del feedback
            coeffs = self.filter_coeffs[ch, :n_taps]
            y_hat = X.dot(coeffs)  # estimación del feedback

            # Error entre referencia y estimación (lo usamos para adaptar)
            error = ref - y_hat

            # Normalización y actualización LMS
            norm = np.sum(X * X) + eps
            self.filter_coeffs[ch, :n_taps] += (self.mu / norm) * X.T.dot(error)

            # Señal limpia = entrada - estimación del feedback
            cleaned[:, ch] = x - y_hat

        # Volver a int16
        cleaned_int16 = np.clip(cleaned * 32768.0, -32768, 32767).astype(np.int16)
        return cleaned_int16

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        self.chunk_number = (self.chunk_number + 1) % self.CHUNK_NUMBERS

        # Actualizar la estimación del feedback con la salida anterior
        if hasattr(self, 'previous_output'):
            self.update_feedback_estimate(self.previous_output)

        # Aplicar filtrado LMS a la señal de entrada (se devuelve int16)
        ADC_processed = self._lms_filter(ADC.copy())

        # Procesamiento normal del buffer
        packed_chunk = super().pack(self.chunk_number, ADC_processed)
        self.send(packed_chunk)
        chunk = super().unbuffer_next_chunk()
        super().play_chunk(DAC, chunk)

        # Guardar la salida actual para la próxima iteración
        self.previous_output = DAC.copy()

        return ADC_processed

    def update_feedback_estimate(self, output_signal):
        """Actualiza la estimación del feedback basado en la salida reciente."""
        # Almacenar la salida en el historial
        self.output_history[self.history_index] = output_signal.astype(np.float32)
        self.history_index = (self.history_index + 1) % len(self.output_history)

        if not hasattr(self, 'feedback_estimate'):
            # Inicializar la estimación en la primera iteración
            self.feedback_estimate = output_signal.astype(np.float32).copy()
        else:
            # Actualizar la estimación de forma adaptativa
            alpha = 0.01  # Tasa de adaptación del filtro
            self.feedback_estimate = (1 - alpha) * self.feedback_estimate + alpha * output_signal.astype(np.float32)

class Feedback_Suppression__verbose(Feedback_Suppression, buffer.Buffering__verbose):
    def __init__(self):
        super().__init__()
        self.feedback_suppression_stats = {
            'signals_processed': 0,
            'feedback_detected_count': 0,
            'average_suppression_db': 0
        }
    
    def spectral_subtraction(self, input_signal, feedback_estimate):
        """Versión verbose que registra estadísticas."""
        output = super().spectral_subtraction(input_signal, feedback_estimate)
        
        # Calcular estadísticas
        input_power = np.mean(input_signal.astype(np.float32)**2)
        output_power = np.mean(output.astype(np.float32)**2)
        
        if input_power > 0 and output_power > 0:
            suppression_db = 10 * np.log10(input_power / output_power)
            self.feedback_suppression_stats['average_suppression_db'] = (
                self.feedback_suppression_stats['average_suppression_db'] * 
                self.feedback_suppression_stats['signals_processed'] + suppression_db
            ) / (self.feedback_suppression_stats['signals_processed'] + 1)
        
        self.feedback_suppression_stats['signals_processed'] += 1
        
        return output
    
    def detect_feedback(self, signal, threshold=0.9):
        detected = super().detect_feedback(signal, threshold)
        if detected:
            self.feedback_suppression_stats['feedback_detected_count'] += 1
        return detected
    
    def print_final_averages(self):
        super().print_final_averages()
        if self.feedback_suppression_stats['signals_processed'] > 0:
            print(f"\nFeedback Suppression Statistics:")
            print(f"Signals processed: {self.feedback_suppression_stats['signals_processed']}")
            print(f"Feedback detected: {self.feedback_suppression_stats['feedback_detected_count']} times")
            print(f"Average suppression: {self.feedback_suppression_stats['average_suppression_db']:.2f} dB")

try:
    import argcomplete  # <tab> completion for argparse.
except ImportError:
    logging.warning("Unable to import argcomplete (optional)")

if __name__ == "__main__":
    minimal.parser.description = __doc__
    try:
        argcomplete.autocomplete(minimal.parser)
    except Exception:
        logging.warning("argcomplete not working :-/")
    minimal.args = minimal.parser.parse_known_args()[0]

    if minimal.args.show_stats or minimal.args.show_samples or minimal.args.show_spectrum:
        intercom = Feedback_Suppression__verbose()
    else:
        intercom = Feedback_Suppression()
        
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()