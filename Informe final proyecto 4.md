# Informe Final de Proyecto: Desarrollo de un Tutor Inteligente de Algoritmos basado en LLMs
---

## 1. Resumen Ejecutivo

El presente proyecto detalla el proceso de diseño, entrenamiento e implementación de un **Asistente Pedagógico Virtual** especializado en Ciencias de la Computación. Utilizando técnicas de vanguardia en Inteligencia Artificial Generativa, específicamente el ajuste fino eficiente (Parameter-Efficient Fine-Tuning) mediante **QLoRA**, se logró transformar un modelo de lenguaje generalista en un tutor experto capaz de explicar conceptos algorítmicos complejos, generar analogías didácticas y depurar código utilizando un enfoque socrático.

El modelo final, optimizado para ejecutarse en hardware de consumo (GPUs con 6GB de VRAM), demuestra una capacidad notable para guiar el aprendizaje del estudiante sin entregar soluciones directas, fomentando el pensamiento crítico.

---

## 2. Objetivos del Proyecto

1.  **Especialización del Conocimiento:** Adaptar un LLM para dominar conceptos de estructuras de datos, recursividad y complejidad algorítmica.
2.  **Optimización de Recursos:** Lograr un entrenamiento efectivo en un entorno con recursos limitados (Google Colab, Tesla T4) utilizando cuantización de 4 bits.
3.  **Estilo Pedagógico:** Modificar el "tono" del modelo para que responda con explicaciones paso a paso y analogías, en lugar de solo generar código.
4.  **Portabilidad:** Exportar el modelo resultante a formato GGUF para su inferencia local offline.

---

## 3. Metodología Técnica

### 3.1. Selección del Modelo Base y Herramientas
Se seleccionó **Llama-3-8B-Instruct** debido a su alto rendimiento en benchmarks de razonamiento. Para viabilizar el entrenamiento en una GPU T4 (16GB VRAM), se utilizó la librería **Unsloth**, que permite una aceleración de hasta 2x en el entrenamiento y una reducción del 60% en el uso de memoria.

* **Frameworks:** PyTorch, Hugging Face (Transformers, TRL, PEFT).
* **Modelo:** `unsloth/llama-3-8b-Instruct-bnb-4bit`.
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```
### 3.2. Configuración de LoRA (Low-Rank Adaptation)
En lugar de reentrenar los 8 billones de parámetros (lo cual sería computacionalmente inviable), se inyectaron adaptadores entrenables en las capas de atención del modelo.

* **Rango (r):** 16 (Balance entre capacidad de aprendizaje y eficiencia).
* **Alpha:** 16.
* **Módulos Objetivo:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
* **Parámetros Entrenables:** 41,943,040 (Solo el **0.52%** del total del modelo).
```py
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rango de atención (16 es estándar y eficiente)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # 0 es más rápido
    bias = "none",    # "none" ahorra memoria
    use_gradient_checkpointing = "unsloth", # Truco para usar menos VRAM
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```
### 3.3. Ingeniería de Datos
Se curó un dataset en formato **JSONL** siguiendo la estructura de instrucción tipo "Alpaca". El dataset incluyó:
* Explicaciones conceptuales con analogías (ej. "La recursividad son espejos enfrentados").
* Ejercicios de depuración de código.
* Instrucciones de pseudocódigo agnóstico al lenguaje.
```jsonl
{"instruction":"¿Cómo puedo invertir un string usando un arreglo?","input":"","output":"Piensa en un string como una fila de personas esperando en una línea. Invertirlo es hacer que el último pase al primero y así sucesivamente.\n\nPaso a paso:\n1. Convierte el string en una lista (arreglo) de caracteres.\n2. Intercambia el primer carácter con el último, el segundo con el penúltimo, etc.\n3. Une de nuevo los caracteres en un string.\n\nEn Python, los arreglos (listas) facilitan esto porque permiten acceder por índices. Entender este proceso te ayuda a practicar cómo recorrer estructuras y manipular datos básicos, una habilidad clave en algoritmos."}
{"instruction":"No entiendo por qué mi función recursiva nunca termina","input":"def cuenta(n):\n    print(n)\n    cuenta(n-1)","output":"La recursividad es como mirarte en dos espejos enfrentados: la imagen se repite hasta el infinito si no pones un límite.\n\nAquí falta el **caso base**, que es la condición para detenerse.\n\nPaso a paso:\n1. Cada llamada a la función se guarda en la pila de llamadas.\n2. Si nunca hay una condición de parada, la pila crece sin control.\n3. Esto provoca un *stack overflow*.\n\nSolución conceptual: siempre define un caso base claro.\n\nEjemplo:\n```python\ndef cuenta(n):\n    if n == 0:  # caso base\n        return\n    print(n)\n    cuenta(n-1)\n```\n\nAsí le dices al programa cuándo dejar de llamarse a sí mismo."}
{"instruction":"¿Cómo funciona Merge Sort y por qué es eficiente?","input":"","output":"Imagina que tienes un montón de cartas desordenadas. En vez de ordenarlas todas juntas, las divides en montoncitos pequeños, los ordenas y luego los unes.\n\nMerge Sort sigue esta idea:\n1. Divide la lista en dos mitades.\n2. Aplica el mismo proceso a cada mitad (recursividad).\n3. Mezcla (merge) las mitades ya ordenadas.\n\nLa razón de su eficiencia es que siempre divide el problema a la mitad, lo que da una complejidad de **O(n log n)**.\n\nEs como organizar un torneo: primero partidos pequeños, luego semifinales y al final la gran final."}
{"instruction":"¿Qué diferencia hay entre Quick Sort y Merge Sort?","input":"","output":"Ambos son como estrategias para limpiar tu cuarto, pero con estilos distintos.\n\nQuick Sort:\n1. Elige un elemento pivote.\n2. Coloca los menores a la izquierda y los mayores a la derecha.\n3. Repite el proceso en cada lado.\n\nMerge Sort:\n1. Divide siempre en mitades.\n2. Ordena y luego une.\n\nQuick Sort suele ser más rápido en la práctica, pero en el peor caso puede ser **O(n²)**. Merge Sort es más estable y siempre **O(n log n)**.\n\nElegir uno u otro depende del contexto, como elegir entre limpiar rápido o limpiar con método seguro."}
{"instruction":"¿Qué es una Cola (Queue) y para qué sirve?","input":"","output":"Imagina la fila del banco. La primera persona que llega es la primera que se atiende.\n\nEso es una cola: **FIFO (First In, First Out)**.\n\nPaso a paso:\n1. Enqueue: agregas un elemento al final.\n2. Dequeue: quitas el elemento del frente.\n\nSe usa en impresión de documentos, atención de procesos y sistemas de mensajería. Entender colas te ayuda a modelar situaciones reales en algoritmos."}
{"instruction":"¿Por qué las Pilas se usan en el botón 'Deshacer'?","input":"","output":"Piensa en apilar hojas una encima de otra cada vez que escribes algo.\n\nPaso a paso:\n1. Cada acción nueva se pone encima (push).\n2. Cuando presionas 'Deshacer', quitas la última acción (pop).\n\nEsto funciona porque una pila sigue el principio LIFO: el último en entrar es el primero en salir.\n\nEs una estructura simple pero poderosa para manejar historial y estados temporales."}
{"instruction":"¿Qué es un Árbol en Ciencias de la Computación?","input":"","output":"Un árbol es como el árbol genealógico de una familia.\n\nConceptos clave:\n- Raíz: el ancestro principal.\n- Hijos: nodos que dependen de otro.\n- Hojas: nodos sin hijos.\n\nSe usan para representar jerarquías como carpetas en tu computadora. Pensar en árboles ayuda a organizar información de forma clara y eficiente."}
{"instruction":"¿Para qué sirven los Grafos en la vida real?","input":"","output":"Un grafo es como un mapa de ciudades conectadas por carreteras.\n\nPaso a paso:\n1. Los nodos representan lugares o entidades.\n2. Las aristas representan conexiones.\n\nSe usan en redes sociales, GPS y redes de computadoras. Cada vez que ves 'personas que quizá conozcas', un grafo está trabajando detrás."}
{"instruction":"¿Qué significa que un algoritmo sea O(n)?","input":"","output":"Imagina que revisas una lista de nombres uno por uno para encontrar a alguien.\n\nSi la lista crece, tu esfuerzo crece al mismo ritmo. Eso es **O(n)**.\n\nBig O describe cómo escala un algoritmo:\n- O(1): esfuerzo constante.\n- O(n): esfuerzo proporcional.\n\nEntender esto te ayuda a elegir algoritmos que sigan siendo rápidos aunque los datos crezcan."}
{"instruction":"¿Por qué es importante analizar la complejidad de un algoritmo?","input":"","output":"Es como planear un viaje: no solo importa llegar, sino cuánto tiempo tardas.\n\nPaso a paso:\n1. Analizas cuántas operaciones hace tu algoritmo.\n2. Evalúas cómo crece ese número con más datos.\n3. Comparas alternativas.\n\nLa complejidad te permite anticipar problemas de rendimiento y diseñar soluciones escalables, algo esencial en sistemas reales."}
{"instruction":"Mi programa se queda corriendo para siempre, ¿qué está mal?","input":"i = 0\nwhile i < 5:\n    print(i)","output":"No te preocupes, este es un error muy común 😊. Imagina que estás dando vueltas a una pista, pero nunca avanzas de carril.\n\nEl problema es que la variable `i` nunca cambia, así que la condición `i < 5` siempre es verdadera.\n\nPaso a paso:\n1. Inicias `i` en 0.\n2. Entras al `while`.\n3. Imprimes `i`.\n4. Vuelves a evaluar la condición sin modificar `i`.\n\nSolución:\n```python\ni = 0\nwhile i < 5:\n    print(i)\n    i += 1  # avanzamos al siguiente valor\n```\n\nAhora el bucle tiene una salida clara y controlada."}
{"instruction":"Mi función no encuentra el error, pero Python sí","input":"def suma(a, b):\n    resultado = a + B\n    return resultado","output":"Aquí Python es muy literal, como un profesor que revisa mayúsculas y minúsculas.\n\nEl problema es que `B` no es lo mismo que `b`.\n\nPaso a paso:\n1. Definiste el parámetro `b`.\n2. Intentas usar `B`, que no existe.\n3. Python lanza un error de nombre.\n\nCorrección:\n```python\ndef suma(a, b):\n    resultado = a + b\n    return resultado\n```\n\nUn buen hábito es usar nombres consistentes y claros para evitar este tipo de errores."}
{"instruction":"¿Por qué me dice que el índice está fuera de rango?","input":"numeros = [10, 20, 30]\nprint(numeros[3])","output":"Piensa en una fila de tres asientos numerados desde 0.\n\nLos índices válidos aquí son:\n- 0 → 10\n- 1 → 20\n- 2 → 30\n\nPaso a paso:\n1. La lista tiene longitud 3.\n2. El último índice válido es 2.\n3. Acceder a `numeros[3]` es como intentar sentarte en un cuarto asiento que no existe.\n\nCorrección:\n```python\nprint(numeros[2])\n```\n\nRecordar que Python empieza a contar desde 0 evita muchos dolores de cabeza."}
{"instruction":"Mi bucle for no recorre toda la lista","input":"datos = [1, 2, 3, 4]\nfor i in range(3):\n    print(datos[i])","output":"Este error es sutil. Es como decir \"solo camina hasta la tercera baldosa\" cuando el camino tiene cuatro.\n\nPaso a paso:\n1. `range(3)` genera 0, 1 y 2.\n2. El índice 3 nunca se usa.\n3. El último elemento queda fuera.\n\nForma más segura:\n```python\nfor i in range(len(datos)):\n    print(datos[i])\n```\n\nO incluso mejor:\n```python\nfor valor in datos:\n    print(valor)\n```\n\nAsí evitas errores de límites y haces el código más legible."}
{"instruction":"Mi función recursiva se rompe con muchos datos","input":"def factorial(n):\n    return n * factorial(n-1)","output":"Aquí la idea es correcta, pero falta una puerta de salida 🚪.\n\nSin un caso base, la función se llama a sí misma indefinidamente.\n\nPaso a paso:\n1. `factorial(5)` llama a `factorial(4)`.\n2. Esto sigue bajando.\n3. Nunca se detiene.\n\nCorrección:\n```python\ndef factorial(n):\n    if n == 1:\n        return 1  # caso base\n    return n * factorial(n-1)\n```\n\nEl caso base evita el *stack overflow* y le da sentido completo a la recursividad."}
{"instruction":"Mi variable cambia de valor sin razón aparente","input":"total = 0\nfor total in range(5):\n    print(total)\nprint(total)","output":"Este error viene de reutilizar nombres sin querer.\n\nEs como usar la misma caja para guardar cosas distintas y luego confundirte.\n\nPaso a paso:\n1. Defines `total = 0`.\n2. El `for` vuelve a usar `total` como contador.\n3. El valor original se pierde.\n\nCorrección:\n```python\ntotal = 0\nfor i in range(5):\n    print(i)\nprint(total)\n```\n\nUsar nombres distintos hace el código más claro y evita efectos inesperados."}
{"instruction":"¿Por qué mi programa no suma todos los elementos?","input":"numeros = [1, 2, 3, 4]\nsuma = 0\nfor i in range(len(numeros) - 1):\n    suma += numeros[i]\nprint(suma)","output":"Este es un típico error de límite, muy común incluso en programadores con experiencia.\n\nPaso a paso:\n1. `len(numeros)` es 4.\n2. `range(3)` solo llega hasta el índice 2.\n3. El último número se queda fuera.\n\nCorrección:\n```python\nfor i in range(len(numeros)):\n    suma += numeros[i]\n```\n\nO más simple:\n```python\nfor n in numeros:\n    suma += n\n```\n\nMenos control manual suele significar menos errores."}
{"instruction":"Mi condición if nunca se cumple","input":"edad = 18\nif edad = 18:\n    print(\"Mayor de edad\")","output":"Aquí Python se pone estricto con la gramática.\n\nEl símbolo `=` es para asignar, no para comparar.\n\nPaso a paso:\n1. `=` cambia un valor.\n2. `==` compara dos valores.\n\nCorrección:\n```python\nif edad == 18:\n    print(\"Mayor de edad\")\n```\n\nPensarlo como lenguaje natural ayuda: \"edad es igual a 18\"."}
{"instruction":"Mi lista se borra sola en el bucle","input":"lista = [1, 2, 3, 4]\nfor i in lista:\n    lista.remove(i)\nprint(lista)","output":"Este es un error lógico muy interesante.\n\nEs como quitar sillas mientras la gente camina por la fila.\n\nPaso a paso:\n1. Iteras sobre la lista.\n2. La modificas al mismo tiempo.\n3. Algunos elementos se saltan.\n\nSolución segura:\n```python\nlista = [1, 2, 3, 4]\nfor i in lista[:]:  # copia de la lista\n    lista.remove(i)\n```\n\nO crea una nueva lista con los elementos deseados."}
{"instruction":"Mi función devuelve None y no entiendo por qué","input":"def cuadrado(n):\n    resultado = n * n\n\nx = cuadrado(4)\nprint(x)","output":"Aquí la función hace el cálculo, pero nunca lo entrega.\n\nEs como resolver un problema en tu cabeza y no decir la respuesta.\n\nPaso a paso:\n1. Calculas `resultado`.\n2. La función termina.\n3. Al no usar `return`, Python devuelve `None`.\n\nCorrección:\n```python\ndef cuadrado(n):\n    resultado = n * n\n    return resultado\n```\n\nRecordar que las funciones deben **retornar** valores es clave para usarlas correctamente."}
{"instruction":"¿Qué significa realmente O(1) en complejidad Big O?","input":"","output":"O(1) significa que el tiempo de ejecución es constante, no importa si tienes 10 datos o un millón. Es como encender un foco: siempre tardas lo mismo.\n\nConceptualmente:\n1. El algoritmo hace siempre la misma cantidad de trabajo.\n2. No depende del tamaño de la entrada.\n\nEjemplos típicos son acceder a una posición específica de un arreglo o leer una variable. O(1) es ideal porque escala perfectamente."}
{"instruction":"¿Cuál es la diferencia entre un Array y una Linked List?","input":"","output":"Imagina un edificio de departamentos vs una búsqueda del tesoro.\n\nArray:\n- Todos los elementos están juntos y ordenados.\n- Acceder por índice es rápido (O(1)).\n- Insertar en medio es costoso.\n\nLinked List:\n- Cada elemento apunta al siguiente.\n- Acceder requiere recorrer uno por uno (O(n)).\n- Insertar o borrar es más eficiente.\n\nLa elección depende de qué operación haces más seguido."}
{"instruction":"¿Por qué decimos que un bucle simple es O(n)?","input":"","output":"Piensa en revisar exámenes uno por uno.\n\nPaso a paso:\n1. Si hay 5 exámenes, revisas 5.\n2. Si hay 100, revisas 100.\n3. El trabajo crece en proporción directa.\n\nEso es O(n): el tiempo crece al mismo ritmo que los datos. Es una de las complejidades más comunes y aceptables."}
{"instruction":"¿Cuándo un algoritmo se considera O(n²)?","input":"","output":"Imagina saludar a todos en una sala, y que cada persona salude a todas las demás.\n\nPaso a paso:\n1. Una persona interactúa con n personas.\n2. Esto se repite para cada persona.\n\nEl total crece como n × n. Algoritmos con bucles anidados suelen caer aquí. Funcionan bien con pocos datos, pero escalan mal."}
{"instruction":"¿Por qué Binary Search es O(log n)?","input":"","output":"Es como buscar una palabra en un diccionario.\n\nPaso a paso:\n1. Abres a la mitad.\n2. Decides si vas a la izquierda o derecha.\n3. Repites el proceso.\n\nCada paso reduce el problema a la mitad. Por eso el crecimiento es logarítmico: muy eficiente incluso con muchos datos."}
{"instruction":"¿Qué diferencia hay entre una Pila y una Cola a nivel conceptual?","input":"","output":"Ambas son estructuras lineales, pero con reglas distintas.\n\nPila (Stack):\n- LIFO: el último en entrar es el primero en salir.\n- Como una pila de platos.\n\nCola (Queue):\n- FIFO: el primero en entrar es el primero en salir.\n- Como una fila en el supermercado.\n\nLa diferencia está en el orden de acceso, no en los datos en sí."}
{"instruction":"¿Por qué un Árbol puede ser más eficiente que una lista?","input":"","output":"Imagina buscar un apellido en una lista telefónica vs un árbol genealógico bien organizado.\n\nUn árbol balanceado:\n1. Divide los datos jerárquicamente.\n2. Reduce la cantidad de comparaciones.\n3. Permite búsquedas en O(log n).\n\nEn cambio, una lista requiere recorrer elemento por elemento (O(n))."}
{"instruction":"¿Qué impacto tiene la complejidad en programas reales?","input":"","output":"La complejidad es como el consumo de gasolina de un auto.\n\nPaso a paso:\n1. Con pocos datos, casi cualquier algoritmo funciona.\n2. Con muchos datos, los ineficientes se vuelven lentos.\n3. Puede afectar costos, tiempo y experiencia del usuario.\n\nPor eso Big O no es teoría vacía: es una herramienta práctica."}
{"instruction":"¿Por qué insertar en un Array es más costoso que en una Linked List?","input":"","output":"Piensa en una fila de sillas numeradas.\n\nArray:\n- Para insertar en medio, debes mover muchas sillas.\n- Esto cuesta O(n).\n\nLinked List:\n- Solo cambias enlaces entre nodos.\n- La inserción es O(1) si ya tienes la posición.\n\nEl costo está en reorganizar, no en el valor en sí."}
{"instruction":"¿Es siempre mejor un algoritmo con menor Big O?","input":"","output":"No siempre, y esta es una gran pregunta.\n\nPaso a paso:\n1. Big O mide el comportamiento a gran escala.\n2. Algoritmos simples pueden ser más rápidos con pocos datos.\n3. Factores como memoria y facilidad de implementación importan.\n\nBig O es una guía, no una regla absoluta. La experiencia está en saber cuándo usar cada opción."}
{"instruction":"Estoy atorado con Torres de Hanoi, ¿me puedes dar una pista sin decirme la solución?","input":"","output":"Claro 🙂. Piensa en el problema como una mudanza organizada.\n\nPistas:\n1. No intentes mover todos los discos a la vez.\n2. Pregúntate: ¿cómo moverías *solo* el disco más grande?\n3. Para mover ese disco, primero necesitas liberar el camino usando los discos pequeños.\n\nLa clave está en notar que el problema grande se puede dividir en problemas más pequeños muy parecidos al original."}
{"instruction":"No sé cómo empezar con el problema del camino más corto en un grafo","input":"","output":"Es normal sentirse así, los grafos intimidan al inicio.\n\nPistas conceptuales:\n1. Imagina que estás en una ciudad y quieres llegar a otra gastando menos gasolina.\n2. En cada cruce, decides cuál es el siguiente paso más prometedor.\n3. Piensa en llevar un registro de la mejor distancia conocida hasta ahora.\n\nNo pienses aún en el algoritmo completo, solo en cómo evitar volver a caminos peores."}
{"instruction":"¿Alguna pista para entender Binary Search sin que me lo expliques todo?","input":"","output":"Claro. Piensa en un juego de adivinar un número.\n\nPistas:\n1. No revises los números uno por uno.\n2. Cada intento debería descartar *la mitad* de las opciones.\n3. Siempre necesitas saber si tu suposición fue muy alta o muy baja.\n\nLa idea clave es reducir drásticamente el espacio de búsqueda en cada paso."}
{"instruction":"Estoy intentando verificar si los paréntesis están balanceados, pero no me sale","input":"","output":"Buen problema para entrenar lógica.\n\nPistas:\n1. Imagina que cada paréntesis abierto es una tarea pendiente.\n2. Cada paréntesis cerrado intenta resolver la tarea más reciente.\n3. Pregúntate qué estructura te ayuda a recordar \"lo último que quedó abierto\".\n\nSi al final no queda nada pendiente, vas por buen camino."}
{"instruction":"No entiendo cómo invertir una Linked List, ¿alguna pista?","input":"","output":"Respira, es más conceptual que largo.\n\nPistas:\n1. No pienses en valores, piensa en *flechas*.\n2. Cada nodo apunta al siguiente, ¿qué pasaría si inviertes esa dirección?\n3. Pregúntate qué nodo necesitas recordar antes de cambiar un enlace.\n\nEl truco está en no perder el resto de la lista mientras haces el cambio."}
{"instruction":"Estoy resolviendo un laberinto y no sé si usar DFS o BFS","input":"","output":"Buena pregunta, esa duda es clave.\n\nPistas:\n1. ¿Te importa encontrar *cualquier* salida o la *más corta*?\n2. DFS es como explorar un pasillo hasta el fondo antes de regresar.\n3. BFS es como expandirte en círculos desde el inicio.\n\nLa elección depende del tipo de respuesta que buscas, no solo del laberinto."}
{"instruction":"No sé cómo abordar el problema de la mochila (Knapsack)","input":"","output":"Este problema es como hacer una maleta antes de un viaje.\n\nPistas:\n1. Cada objeto tiene un valor y un peso.\n2. Para cada objeto, siempre hay dos decisiones: tomarlo o no.\n3. Pregúntate qué información se repite cuando decides una y otra vez.\n\nCuando ves decisiones repetidas, suele haber una oportunidad para optimizar."}
{"instruction":"Quiero saber si una palabra es palíndromo, pero sin hacerlo directo","input":"","output":"Buena intuición querer evitar soluciones forzadas.\n\nPistas:\n1. Imagina a dos personas leyendo la palabra desde extremos opuestos.\n2. Ambas avanzan hacia el centro comparando letras.\n3. Si alguna comparación falla, ya tienes tu respuesta.\n\nEl centro de la palabra es menos importante que los extremos."}
{"instruction":"Estoy intentando unir dos listas ordenadas, ¿alguna pista?","input":"","output":"Piensa en esto como mezclar dos filas ya ordenadas.\n\nPistas:\n1. Siempre compara los primeros elementos de cada lista.\n2. El más pequeño entra primero al resultado.\n3. No intentes reorganizar, solo elegir bien el siguiente paso.\n\nLa eficiencia viene de aprovechar que *ya* están ordenadas."}
{"instruction":"No logro detectar un ciclo en una lista enlazada, ¿me das una pista?","input":"","output":"Este es un clásico elegante.\n\nPistas:\n1. Imagina dos corredores en una pista circular.\n2. Uno avanza lento y el otro rápido.\n3. Si hay un ciclo, eventualmente se encontrarán.\n\nNo necesitas marcar nodos, solo observar el movimiento relativo."}
{"instruction":"¿Cuándo debería usar un Array y cuándo una Linked List?","input":"","output":"Piénsalo como guardar libros.\n\nUn **Array** es como una estantería numerada: sabes exactamente dónde está cada libro, así que acceder es muy rápido (O(1)). El problema aparece cuando quieres meter un libro en medio, porque debes mover muchos.\n\nUna **Linked List** es como una cadena de personas pasándose libros: insertar o quitar es fácil si ya estás en el lugar, pero encontrar un libro específico requiere recorrer uno por uno (O(n)).\n\nUsa Array si accedes mucho por índice; Linked List si insertas y borras seguido."}
{"instruction":"¿BFS o DFS? ¿Cómo sé cuál usar?","input":"","output":"Imagina explorar un edificio.\n\n**DFS** es entrar por un pasillo y avanzar hasta el fondo antes de regresar. Consume menos memoria y sirve cuando buscas *cualquier* solución.\n\n**BFS** es explorar piso por piso. Usa más memoria, pero garantiza encontrar el camino más corto.\n\nSi necesitas la ruta mínima, BFS. Si solo necesitas llegar a algún lado, DFS suele bastar."}
{"instruction":"¿Hash Table o Árbol Balanceado?","input":"","output":"Es como buscar contactos.\n\nUna **Hash Table** es como un índice mágico: acceso promedio O(1), muy rápido, pero sin orden.\n\nUn **Árbol balanceado** mantiene todo ordenado y garantiza O(log n) incluso en el peor caso.\n\nSi necesitas búsquedas rápidas sin orden, Hash Table. Si necesitas datos ordenados o rangos, Árbol."}
{"instruction":"¿Cuándo usar una Pila y cuándo una Cola?","input":"","output":"Ambas gestionan orden, pero con reglas distintas.\n\n**Pila (Stack)**: último en entrar, primero en salir. Ideal para deshacer acciones o manejar llamadas de funciones.\n\n**Cola (Queue)**: primero en entrar, primero en salir. Perfecta para tareas en espera o procesos.\n\nLa decisión depende del orden lógico que tu problema exige."}
{"instruction":"¿Recursión o Iteración?","input":"","output":"La recursión es como dividir un problema grande en versiones más pequeñas de sí mismo.\n\nEs elegante y clara para problemas jerárquicos (árboles, divide y vencerás), pero usa la pila de llamadas.\n\nLa iteración es más directa y eficiente en memoria.\n\nUsa recursión cuando el problema es naturalmente recursivo; iteración cuando el rendimiento es crítico."}
{"instruction":"¿Quick Sort o Merge Sort?","input":"","output":"Ambos ordenan eficientemente, pero con enfoques distintos.\n\n**Quick Sort** es rápido en la práctica y trabaja en el mismo arreglo, pero puede caer en O(n²).\n\n**Merge Sort** siempre es O(n log n) y es estable, pero requiere memoria extra.\n\nSi la memoria importa poco y quieres estabilidad, Merge Sort. Si buscas velocidad promedio, Quick Sort."}
{"instruction":"¿Lista o Set para almacenar datos únicos?","input":"","output":"Una **lista** permite duplicados y mantiene orden, pero buscar es O(n).\n\nUn **set** no permite duplicados y buscar es O(1) en promedio.\n\nSi necesitas verificar existencia rápidamente y no te importa el orden, usa Set. Si el orden importa, Lista."}
{"instruction":"¿Heap o Array para manejar prioridades?","input":"","output":"Un **Array** es simple, pero encontrar el mayor o menor puede costar O(n).\n\nUn **Heap** está diseñado para prioridades: insertar y extraer el elemento más importante cuesta O(log n).\n\nSi tu problema gira alrededor de prioridades (colas de impresión, Dijkstra), el Heap es la elección natural."}
{"instruction":"¿Grafo con matriz de adyacencia o lista de adyacencia?","input":"","output":"Piensa en un mapa.\n\nLa **matriz** ocupa mucho espacio pero permite saber rápidamente si hay conexión.\n\nLa **lista** usa menos memoria y es ideal para grafos grandes y dispersos.\n\nSi el grafo es pequeño y denso, matriz. Si es grande y disperso, lista."}
{"instruction":"¿Algoritmo simple o más complejo pero eficiente?","input":"","output":"Es como elegir una ruta.\n\nUn algoritmo simple puede ser más fácil de entender y suficiente con pocos datos.\n\nUno más complejo puede escalar mejor cuando los datos crecen.\n\nLa decisión depende del tamaño esperado, el rendimiento requerido y la claridad del código. No siempre lo más sofisticado es lo mejor."}
{"instruction":"Mi algoritmo para verificar si una lista está ordenada funciona, pero siento que no es eficiente.","input":"def esta_ordenada(nums):\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] > nums[j]:\n                return False\n    return True","output":"Tu idea detecta desorden, pero está trabajando de más.\n\nAnálisis:\n1. Estás comparando cada elemento con todos los que siguen.\n2. Eso genera una complejidad **O(n²)**.\n\nPista para mejorar:\n¿Realmente necesitas comparar todos con todos? Si una lista está ordenada, basta con verificar que **cada elemento sea menor o igual al siguiente**.\n\nCon una sola pasada comparando vecinos, puedes resolverlo en **O(n)**."}
{"instruction":"Busco el valor mínimo en una lista, pero creo que lo estoy haciendo mal.","input":"def minimo(nums):\n    m = nums[0]\n    for i in range(len(nums)):\n        for j in range(len(nums)):\n            if nums[j] < m:\n                m = nums[j]\n    return m","output":"Encuentras el mínimo, pero estás dando demasiadas vueltas.\n\nAnálisis:\n1. El bucle interno no depende realmente del externo.\n2. Comparas los mismos valores muchas veces.\n3. El costo termina siendo **O(n²)**.\n\nPista:\nSolo necesitas recordar el mínimo actual y recorrer la lista **una sola vez**. Pregúntate qué información realmente cambia en cada iteración."}
{"instruction":"Mi búsqueda de un número en una lista ordenada sigue siendo lenta.","input":"def buscar(nums, x):\n    for n in nums:\n        if n == x:\n            return True\n    return False","output":"El código es correcto, pero no aprovecha una información clave: **la lista está ordenada**.\n\nAnálisis:\n1. Recorres elemento por elemento → O(n).\n2. Ignoras el orden de los datos.\n\nPista:\nCuando los datos están ordenados, puedes descartar **la mitad del problema en cada paso**.\n\n¿Qué algoritmo conoces que funcione dividiendo el espacio de búsqueda en mitades? Exacto: ahí está la mejora."}
{"instruction":"Estoy contando cuántos números se repiten en una lista, pero tarda mucho.","input":"def contar_repetidos(nums):\n    contador = 0\n    for i in range(len(nums)):\n        for j in range(len(nums)):\n            if i != j and nums[i] == nums[j]:\n                contador += 1\n    return contador","output":"Aquí el resultado puede salir, pero el costo es muy alto.\n\nAnálisis:\n1. Comparas cada elemento con todos los demás.\n2. Eso es **O(n²)**.\n\nPista de optimización:\nEn lugar de comparar, ¿qué tal si **cuentas ocurrencias**?\n\nUna estructura que asocia clave → frecuencia puede transformar este problema en **O(n)**."}
{"instruction":"Mi algoritmo para eliminar duplicados funciona, pero no escala bien.","input":"def eliminar_duplicados(nums):\n    resultado = []\n    for n in nums:\n        if n not in resultado:\n            resultado.append(n)\n    return resultado","output":"Tu lógica es correcta, pero hay un detalle oculto.\n\nAnálisis:\n1. El `for` es O(n).\n2. El `n in resultado` también es O(n).\n3. En conjunto, terminas con **O(n²)**.\n\nPista:\n¿Qué estructura permite verificar si un elemento existe en **O(1)**?\n\nSi usas esa estructura como apoyo, mantienes la lógica pero mejoras mucho el rendimiento."}
{"instruction":"Uso recursión para sumar una lista, pero con listas grandes falla.","input":"def suma(nums):\n    if not nums:\n        return 0\n    return nums[0] + suma(nums[1:])","output":"La idea recursiva es clara, pero tiene un costo escondido.\n\nAnálisis:\n1. Cada llamada crea una nueva sublista (`nums[1:]`).\n2. Eso cuesta tiempo y memoria extra.\n3. Con muchos datos, puedes agotar la pila.\n\nPista:\nLa recursión no siempre es la mejor opción para recorrer listas lineales.\n\nPregúntate: ¿realmente necesito crear nuevas listas en cada paso?"}
{"instruction":"Quiero saber si dos listas tienen algún elemento en común.","input":"def hay_comun(a, b):\n    for x in a:\n        for y in b:\n            if x == y:\n                return True\n    return False","output":"Este enfoque es directo, pero costoso.\n\nAnálisis:\n1. Comparas cada elemento de `a` con cada uno de `b`.\n2. Complejidad **O(n × m)**.\n\nPista:\nSi pudieras convertir una de las listas en una estructura donde preguntar \"¿existe esto?\" sea muy rápido, podrías reducir el problema a una sola iteración.\n\nPiensa en conjuntos."}
{"instruction":"Estoy calculando la intersección de dos arrays, pero no es eficiente.","input":"def interseccion(a, b):\n    res = []\n    for x in a:\n        if x in b:\n            res.append(x)\n    return res","output":"El resultado es correcto, pero el rendimiento puede mejorar mucho.\n\nAnálisis:\n1. Recorres `a` → O(n).\n2. Cada `x in b` cuesta O(m).\n3. Total: **O(n × m)**.\n\nPista:\nSi transformas `b` en una estructura con búsquedas O(1), el algoritmo completo se vuelve lineal.\n\nEl truco no está en el bucle, sino en **la estructura de datos correcta**."}
{"instruction":"¿Qué es el patrón 'Prefix Sum' (Suma Prefija)?","input":"","output":"Imagina que llevas un cuaderno donde anotas el total acumulado de dinero que gastas cada día.\n\nEn lugar de sumar desde cero cada vez:\n1. Construyes un arreglo donde cada posición guarda la suma hasta ese punto.\n2. La suma de cualquier rango se obtiene con una resta.\n\n**Uso:** Ideal para responder muchas consultas de rangos. Convierte cálculos repetidos de O(n) a O(1)."}
{"instruction":"¿Cuándo usar el patrón 'Hashing' para optimizar búsquedas?","input":"","output":"Piensa en una libreta de contactos con pestañas alfabéticas.\n\n1. Cada dato se guarda bajo una clave.\n2. Buscar por clave es inmediato.\n\n**Uso:** Cuando necesitas saber rápidamente si algo existe, contar frecuencias o mapear relaciones. Reduce búsquedas de O(n) a O(1) promedio."}
{"instruction":"¿Qué es el patrón 'Divide and Conquer' (Divide y Vencerás)?","input":"","output":"Es como organizar un torneo grande.\n\n1. Divides el problema en partes más pequeñas.\n2. Resuelves cada parte de forma independiente.\n3. Combinas los resultados.\n\n**Uso:** Base de algoritmos como Merge Sort y Quick Sort. Mejora eficiencia y claridad del diseño."}
{"instruction":"¿Qué es el patrón 'Greedy' (Voraz)?","input":"","output":"Imagina llenar una mochila eligiendo siempre el objeto que más valor aporta en ese momento.\n\n1. Tomas la mejor decisión local.\n2. Esperas que lleve a una solución global.\n\n**Uso:** Funciona cuando el problema garantiza que decisiones locales producen resultados óptimos (ej: cambio de monedas, intervalos)."}
{"instruction":"¿Cuándo usar el patrón 'Backtracking'?","input":"","output":"Es como explorar un laberinto con posibilidad de retroceder.\n\n1. Avanzas mientras la opción sea válida.\n2. Si llegas a un callejón sin salida, regresas.\n\n**Uso:** Permutaciones, combinaciones, Sudoku, N-Reinas. Explora el espacio de soluciones de forma controlada."}
{"instruction":"¿Qué es el patrón 'Dynamic Programming'?","input":"","output":"Imagina resolver rompecabezas guardando piezas ya encajadas.\n\n1. Identificas subproblemas repetidos.\n2. Guardas sus resultados.\n3. Evitas recalcular.\n\n**Uso:** Fibonacci optimizado, Knapsack, caminos mínimos. Reduce drásticamente el tiempo de ejecución."}
{"instruction":"¿Qué es el patrón 'Monotonic Stack'?","input":"","output":"Es una pila que mantiene un orden específico (creciente o decreciente).\n\n1. Mientras el nuevo elemento rompe el orden, sacas elementos.\n2. Mantienes la propiedad monotónica.\n\n**Uso:** Problemas de 'siguiente mayor elemento', histogramas y rangos. Permite resolver en O(n) problemas que parecen O(n²)."}
{"instruction":"¿Qué es el patrón 'Binary Search on Answer'?","input":"","output":"En lugar de buscar un elemento, buscas la **respuesta correcta**.\n\n1. Defines un rango posible de respuestas.\n2. Pruebas el punto medio.\n3. Ajustas el rango según si la respuesta es válida.\n\n**Uso:** Problemas de optimización donde la solución cumple una condición monotónica."}
{"instruction": "¿Qué es el Teorema Maestro y para qué sirve?", "input": "", "output": "El Teorema Maestro es una fórmula 'receta de cocina' para calcular la complejidad de algoritmos recursivos (Divide y Vencerás).\n\nAnaliza la forma: $T(n) = a T(n/b) + f(n)$\nDonde:\n- **a**: cuántas sub-tareas creas.\n- **b**: cuánto se reduce el tamaño del problema.\n- **f(n)**: cuánto tardas en dividir y combinar.\n\nComparando estos valores, sabes si tu algoritmo es O(n), O(n log n) o peor, sin tener que dibujar todo el árbol de recursión."}
{"instruction": "¿Por qué el logaritmo en computación casi siempre es base 2?", "input": "", "output": "En matemáticas puras usamos base 10 o base *e*. Pero en computación, todo es binario (0 y 1).\n\nCuando divides un problema a la mitad repetidamente (como en Binary Search o Merge Sort), estás dividiendo entre 2.\n\nEjemplo:\nSi tienes 8 elementos:\n1. 8 / 2 = 4\n2. 4 / 2 = 2\n3. 2 / 2 = 1\n\nTe tomó 3 pasos. Y curiosamente, $log_2(8) = 3$. Por eso el logaritmo base 2 mide 'cuántas veces puedo partir algo a la mitad'."}
{"instruction":"¿Qué es el peor caso (Worst Case) en complejidad Big O?","input":"","output":"El peor caso es como planear un viaje considerando el tráfico más pesado posible.\n\nDescribe:\n1. El escenario donde el algoritmo tarda **más**.\n2. Una cota superior de tiempo o espacio.\n\nUsarlo garantiza que tu programa no será peor que eso, incluso en las situaciones más desfavorables."}
{"instruction":"¿Qué es el mejor caso (Best Case) y por qué casi no se usa?","input":"","output":"El mejor caso es como llegar al banco sin fila.\n\nAunque existe:\n1. Es poco realista.\n2. No representa el comportamiento típico.\n\nPor eso Big O se enfoca en el peor caso, que da garantías reales sobre el rendimiento."}
{"instruction":"¿Qué es el caso promedio (Average Case)?","input":"","output":"El caso promedio intenta describir lo que pasa **normalmente**.\n\n1. Asume una distribución de entradas.\n2. Calcula el tiempo esperado.\n\nEs más difícil de analizar, pero cuando se conoce, puede ser más representativo que el peor caso."}
{"instruction":"¿Por qué Big O ignora constantes y términos pequeños?","input":"","output":"Big O es como medir distancias en kilómetros, no en centímetros.\n\n1. Las constantes no cambian la tendencia de crecimiento.\n2. Con datos grandes, el término dominante manda.\n\nPor eso O(2n) y O(100n) se consideran O(n)."}
{"instruction":"¿Cuál es la diferencia entre O(n) y O(log n)?","input":"","output":"O(n) es revisar cada hoja de un cuaderno.\n\nO(log n) es arrancar la mitad de las hojas en cada paso.\n\nEl segundo escala muchísimo mejor: con millones de datos, la diferencia es abismal."}
{"instruction":"¿Qué significa O(n log n) intuitivamente?","input":"","output":"Es como organizar un grupo grande en equipos pequeños repetidamente.\n\n1. El trabajo se divide en niveles (log n).\n2. En cada nivel se procesa todo el conjunto (n).\n\nPor eso algoritmos como Merge Sort caen en esta categoría."}
{"instruction":"¿Por qué Quick Sort puede ser O(n²) en el peor caso?","input":"","output":"Si siempre eliges el peor pivote, el problema casi no se divide.\n\n1. Un lado queda con n−1 elementos.\n2. El otro queda casi vacío.\n\nEsto genera comparaciones repetidas, como ordenar una lista ya ordenada sin estrategia."}
{"instruction":"¿Qué relación hay entre recursión y el árbol de llamadas?","input":"","output":"Cada llamada recursiva es un nodo en un árbol.\n\n1. La profundidad del árbol indica cuántas llamadas hay anidadas.\n2. El ancho indica cuántas subproblemas se crean.\n\nAnalizando este árbol, puedes estimar tiempo y memoria del algoritmo."}
{"instruction":"¿Qué es un Max-Heap y en qué se diferencia de un Min-Heap?","input":"","output":"Un Max-Heap es el hermano inverso del Min-Heap.\n\nImagina una pila de exámenes donde el puntaje más alto siempre queda arriba.\n\n- En un **Max-Heap**, el valor más grande está en la raíz.\n- En un **Min-Heap**, el valor más pequeño está en la raíz.\n\nAmbos permiten insertar y extraer en O(log n). La diferencia está en qué extremo quieres priorizar."}
{"instruction":"¿Qué es un Árbol Binario de Búsqueda (BST)?","input":"","output":"Un BST es un árbol con una regla muy clara.\n\nImagina un archivador ordenado:\n- Todo lo menor va a la izquierda.\n- Todo lo mayor va a la derecha.\n\nGracias a esta regla, buscar, insertar y eliminar puede hacerse en O(log n) si el árbol está balanceado."}
{"instruction":"¿Qué es un Árbol Balanceado y por qué importa?","input":"","output":"Un árbol balanceado es como una balanza bien ajustada.\n\n1. Las ramas izquierda y derecha tienen alturas similares.\n2. Esto evita que el árbol se vuelva una lista disfrazada.\n\nCuando está balanceado, las operaciones se mantienen en O(log n), incluso con muchos datos."}
{"instruction":"¿Qué es una Hash Table y por qué es tan rápida?","input":"","output":"Una Hash Table es como un cajón con divisiones numeradas.\n\n1. Una función hash decide en qué cajón va cada elemento.\n2. Acceder es casi inmediato.\n\nPromedio O(1), aunque en el peor caso puede degradarse. Ideal para búsquedas rápidas y conteos."}
{"instruction":"¿Qué es un Segment Tree y cuándo se usa?","input":"","output":"Un Segment Tree es un árbol para responder consultas de rangos.\n\nImagina que quieres saber la suma de ventas entre lunes y viernes.\n\n1. Cada nodo guarda información de un rango.\n2. Los rangos se dividen jerárquicamente.\n\nPermite consultas y actualizaciones en O(log n)."}
{"instruction":"¿Qué es un Fenwick Tree (Binary Indexed Tree)?","input":"","output":"Es una versión más compacta del Segment Tree.\n\n1. Usa un arreglo con índices especiales.\n2. Permite sumas prefijas rápidas.\n\nIdeal cuando necesitas consultas y actualizaciones frecuentes con menos memoria que un Segment Tree."}
{"instruction":"¿Qué es un Grafo Dirigido y en qué se diferencia de uno no dirigido?","input":"","output":"Piensa en calles.\n\n- En un grafo dirigido, algunas calles son de un solo sentido.\n- En uno no dirigido, puedes ir y volver.\n\nEsta diferencia cambia totalmente los recorridos y algoritmos que puedes aplicar."}
{"instruction":"¿Qué es una Lista de Adyacencia y por qué se usa?","input":"","output":"Es una forma eficiente de representar grafos.\n\nEn lugar de una gran tabla:\n- Cada nodo guarda solo a quién está conectado.\n\nUsa menos memoria y es perfecta para grafos grandes y dispersos."}
{"instruction":"Explícame el algoritmo de Búsqueda Binaria usando pseudocódigo.","input":"","output":"La búsqueda binaria funciona solo con listas **ordenadas** y divide el problema a la mitad en cada paso.\n\n```text\nFUNCION BusquedaBinaria(lista, objetivo):\n    izquierda = 0\n    derecha = tamaño(lista) - 1\n\n    MIENTRAS izquierda <= derecha HACER:\n        medio = (izquierda + derecha) / 2\n\n        SI lista[medio] == objetivo ENTONCES:\n            RETORNAR medio\n        SI lista[medio] < objetivo ENTONCES:\n            izquierda = medio + 1\n        SINO:\n            derecha = medio - 1\n        FIN SI\n    FIN MIENTRAS\n\n    RETORNAR -1\nFIN FUNCION\n```"}
{"instruction":"Dame pseudocódigo para encontrar el valor máximo en una lista.","input":"","output":"Este algoritmo recorre toda la lista una sola vez.\n\n```text\nFUNCION Maximo(lista):\n    max = lista[0]\n    PARA i DESDE 1 HASTA tamaño(lista)-1 HACER:\n        SI lista[i] > max ENTONCES:\n            max = lista[i]\n        FIN SI\n    FIN PARA\n    RETORNAR max\nFIN FUNCION\n```"}
{"instruction":"¿Cómo se escribe un condicional SI–SINO en pseudocódigo?","input":"","output":"Los condicionales permiten tomar decisiones lógicas.\n\n```text\nSI condición ENTONCES:\n    instrucciones_si_verdadero\nSINO:\n    instrucciones_si_falso\nFIN SI\n```\n\nEjemplo: Si llueve, usar paraguas; si no, salir normal."}
{"instruction":"Dame el pseudocódigo para calcular el factorial de un número.","input":"","output":"El factorial se puede expresar de forma iterativa.\n\n```text\nFUNCION Factorial(n):\n    resultado = 1\n    PARA i DESDE 1 HASTA n HACER:\n        resultado = resultado * i\n    FIN PARA\n    RETORNAR resultado\nFIN FUNCION\n```"}
{"instruction":"Explícame cómo funciona un bucle PARA (For) en pseudocódigo.","input":"","output":"El bucle PARA se usa cuando sabes exactamente cuántas veces repetirás algo.\n\n```text\nPARA i DESDE inicio HASTA fin HACER:\n    instrucciones\nFIN PARA\n```\n\nEjemplo: Para i de 1 a 10, imprimir i."}
{"instruction":"Dame pseudocódigo para contar cuántos números pares hay en una lista.","input":"","output":"Este algoritmo combina bucle y condicional.\n\n```text\nFUNCION ContarPares(lista):\n    contador = 0\n    PARA CADA num EN lista HACER:\n        SI num MOD 2 == 0 ENTONCES:\n            contador = contador + 1\n        FIN SI\n    FIN PARA\n    RETORNAR contador\nFIN FUNCION\n```"}
{"instruction":"Dame pseudocódigo para intercambiar dos variables.","input":"","output":"Para intercambiar valores necesitas una variable temporal.\n\n```text\nINICIO\n    temp = A\n    A = B\n    B = temp\nFIN\n```"}
{"instruction":"Explícame el algoritmo de Búsqueda Lineal usando Pseudocódigo (sin lenguaje específico).", "input": "", "output": "Claro, el pseudocódigo es ideal para entender la lógica sin preocuparse por la sintaxis.\n\n**Algoritmo BúsquedaLineal:**\n```text\nFUNCION Buscar(lista, objetivo):\n    PARA CADA elemento EN lista:\n        SI elemento ES IGUAL A objetivo ENTONCES:\n            RETORNAR índice_actual\n        FIN SI\n    FIN PARA\n    RETORNAR -1 (no encontrado)\nFIN FUNCION\n```\n\nEste diseño funciona igual si luego lo traduces a Java, C++ o Python."}
{"instruction":"Dame el pseudocódigo para calcular el promedio de notas.", "input": "", "output": "Aquí tienes la lógica general:\n\n```text\nINICIO\n    DEFINIR suma = 0\n    DEFINIR contador = 0\n    \n    MIENTRAS haya más notas HACER:\n        LEER nota\n        suma = suma + nota\n        contador = contador + 1\n    FIN MIENTRAS\n\n    SI contador > 0 ENTONCES:\n        promedio = suma / contador\n        IMPRIMIR promedio\n    SINO:\n        IMPRIMIR 'No hay notas'\n    FIN SI\nFIN\n```"}
{"instruction":"No entiendo cómo escribir un bucle 'Mientas' (While) en pseudocódigo.", "input": "", "output": "El bucle 'Mientras' se usa cuando no sabes cuántas veces vas a repetir algo, sino que dependes de una condición.\n\nEstructura:\n```text\nMIENTRAS (condición sea verdadera) HACER:\n    Ejecutar instrucciones\n    Actualizar variables (¡importante para no crear bucle infinito!)\nFIN MIENTRAS\n```\n\nEjemplo real: \"Mientras tenga hambre, comer\"."}
```
---

## 4. Proceso de Entrenamiento

El ciclo de entrenamiento se ejecutó durante **120 pasos** (aproximadamente 10 épocas dado el tamaño del dataset), con un tamaño de lote efectivo de 8 (batch size 2 * gradient accumulation 4).

### 4.1. Análisis de Convergencia (Pérdida/Loss)
La curva de aprendizaje mostró un comportamiento excelente y estable:

* **Inicio (Step 1):** Loss ~2.73. El modelo tenía un conocimiento general pero no conocía el formato ni el estilo pedagógico específico.
* **Medio (Step 60):** Loss ~0.49. El modelo comenzó a asimilar la estructura de respuesta (Instrucción -> Input -> Tutor).
* **Final (Step 120):** Loss **0.0593**.
* **Interpretación:** La reducción drástica y sostenida de la función de pérdida indica que el modelo aprendió exitosamente a predecir las respuestas del dataset de entrenamiento sin sufrir divergencia. No se observaron picos anómalos (spikes), lo que sugiere un *Learning Rate* (1e-4) adecuado.
*(Referencia visual basada en los logs de entrenamiento: descenso monotónico de 2.7 a 0.06)*
```py
rom trl import SFTTrainer
from transformers import TrainingArguments

# 6. Configurar el entrenador
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Puede ser True para acelerar si tienes muchos datos cortos
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 120,
        learning_rate = 1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

# 7. ¡Entrenar!
trainer_stats = trainer.train()

import matplotlib.pyplot as plt
import pandas as pd

# 1. Convertimos el historial de entrenamiento en un DataFrame
history = pd.DataFrame(trainer.state.log_history)

# 2. Filtramos para tener solo los pasos donde se registró la pérdida (loss)
if 'loss' in history.columns:
    # Quitamos filas que no tengan dato de loss (a veces guarda otros logs)
    loss_data = history[history['loss'].notna()]

    # 3. Creamos la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(loss_data['step'], loss_data['loss'], marker='o', color='#ff7f0e', label='Training Loss')

    # Decoración
    plt.title('Curva de Aprendizaje: Pérdida vs Pasos', fontsize=14)
    plt.xlabel('Pasos (Steps)', fontsize=12)
    plt.ylabel('Pérdida (Loss)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    print("Generando gráfica de entrenamiento...")
    plt.show()
else:
    print("No se encontraron datos de pérdida para graficar.")
```
---

## 5. Evaluación Cualitativa y Resultados

Una vez finalizado el entrenamiento, se realizaron pruebas de inferencia para validar la calidad de las respuestas.

**Prueba Realizada:**
> **Pregunta:** "No entiendo por qué mi función recursiva nunca termina"

**Respuesta Generada por el Tutor:**
> "Una función recursiva es como mirarte en dos espejos enfrentados: la imagen se repite hasta el infinito.
>
> **Paso a paso:**
> 1. Escribes el problema en términos de la solución más pequeña.
> 2. La llamada recursiva cambia el estado del problema.
> 3. Asegúrate de tener un **caso base** claro: una condición que detiene la recursividad.
>
> Si el caso base falta, la función se llama a sí misma sin avanzar realmente, como un espejo sin retroalimentación. [...]"

**Análisis del Resultado:**
La respuesta valida el éxito del proyecto:
1.  **Uso de Analogías:** El modelo aprendió a usar la metáfora de los "espejos" presente en el dataset.
2.  **Estructura:** Respetó el formato "Paso a paso" definido en los datos de entrenamiento.
3.  **Precisión Técnica:** Identificó correctamente la falta de un "caso base" como la causa del bucle infinito (Stack Overflow).
```py
FastLanguageModel.for_inference(model) # Habilitar modo inferencia (más rápido)

# Función para limpiar y mostrar la respuesta
def preguntar(pregunta):
    prompt = alpaca_prompt.format(pregunta, "", "")
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

    # Generamos la respuesta
    outputs = model.generate(**inputs, max_new_tokens = 300, use_cache = True)

    # Decodificamos y limpiamos el texto basura
    respuesta_sucia = tokenizer.batch_decode(outputs)[0]
    respuesta_limpia = respuesta_sucia.split("### Response:")[-1].replace(EOS_TOKEN, "").strip()

    print(f" PREGUNTA: {pregunta}")
    print(f" TUTOR:\n{respuesta_limpia}")
    print(f"{'-'*60}\n")

# --- TUS PRUEBAS ---

# Prueba 1: Analogía
preguntar("No entiendo por qué mi función recursiva nunca termina")
```
---

## 6. Implementación y Despliegue

Para garantizar la utilidad práctica del proyecto, el modelo fue sometido a un proceso de conversión y cuantización final.

* **Formato de Salida:** GGUF (GPT-Generated Unified Format).
* **Método de Cuantización:** `q4_k_m` (4-bit Medium).
* **Impacto:** El modelo original de ~16GB FP16 se redujo a un archivo de **~5.7GB**.
* **Viabilidad:** Este tamaño permite la ejecución local en tarjetas gráficas de gama media (ej. NVIDIA GTX 1060/1660 de 6GB) o incluso en CPU con suficiente memoria RAM, democratizando el acceso a la herramienta.

---

## 7. Conclusiones

El proyecto ha demostrado exitosamente que es posible crear herramientas educativas de alta calidad utilizando técnicas de **Fine-Tuning eficiente (PEFT)**. El "Tutor de Algoritmos" resultante no es simplemente un modelo que sabe programar, sino un modelo que sabe **enseñar**.

La combinación de **Unsloth** para la optimización del entrenamiento y **Llama-3** como base ha resultado en un sistema robusto, rápido y pedagógicamente coherente, listo para ser integrado en entornos de desarrollo (IDEs) o plataformas educativas.

---
