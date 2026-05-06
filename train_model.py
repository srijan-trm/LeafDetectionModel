import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ── Data generators ───────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    shear_range=0.15,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset_split/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    'dataset_split/validation',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Classes:", train_data.class_indices)

# ── Build model ───────────────────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze for phase 1

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),          # prevents overfitting
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ─────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=4,
        restore_best_weights=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'model/best_model.keras',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]

# ── Phase 1: train head only ──────────────────────────────────────
print("\n── Phase 1: training head ──")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)

# ── Phase 2: fine-tune top layers ────────────────────────────────
print("\n── Phase 2: fine-tuning ──")
base_model.trainable = True
for layer in base_model.layers[:-30]:   # freeze all but last 30
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # much lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)

# ── Save ──────────────────────────────────────────────────────────
model.save("model/plant_disease_model.keras")
print("\nFinal val accuracy:", max(history2.history['val_accuracy']))
print("✅ Model trained and saved!")
