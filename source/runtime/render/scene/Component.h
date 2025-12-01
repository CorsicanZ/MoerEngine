#pragma once

#include "Entity.h"

namespace Moer::ECS {

using Index = uint64;

class IComponentManager {
public:
    virtual ~IComponentManager()                     = default;
    virtual void Copy(const IComponentManager& _rhs) = 0;
    virtual void Merge(IComponentManager& _rhs)      = 0;
    virtual void Clear()                             = 0;
    // virtual void                      Serialize(wi::Archive& archive, EntitySerializer& seri)                          = 0;
    // virtual void                      Component_Serialize(Entity entity, wi::Archive& archive, EntitySerializer& seri) = 0;
    virtual void                 Remove(Entity _entity)                       = 0;
    virtual void                 RemoveKeepSorted(Entity _entity)             = 0;
    virtual void                 MoveItem(Index _index_from, Index _index_to) = 0;
    virtual bool                 Contains(Entity _entity) const               = 0;
    virtual Index                GetIndex(Entity _entity) const               = 0;
    virtual Index                GetCount() const                             = 0;
    virtual Entity               GetEntity(Index _index) const                = 0;
    virtual const Array<Entity>& GetEntityArray() const                       = 0;
};

template<typename Component>
class ComponentManager final : public IComponentManager {
public:
    ComponentManager() noexcept                              = default;
    ComponentManager(ComponentManager&&) noexcept            = default;
    ComponentManager& operator=(ComponentManager&&) noexcept = default;
    ~ComponentManager() noexcept                             = default;

    inline void Copy(const IComponentManager& _rhs) override {
        Copy(static_cast<const ComponentManager<Component>&>(_rhs));
    }

    inline void Merge(IComponentManager& _rhs) override {
        Merge(static_cast<ComponentManager<Component>&>(_rhs));
    }

    inline void Clear() override {
        components.clear();
        entities.clear();
        lookup.clear();
    }

    // // Read/Write everything to an archive depending on the archive state
    // inline void Serialize(wi::Archive& archive, EntitySerializer& seri) {
    //     if (archive.IsReadMode()) {
    //         const uint32 prev_count = components.size();

    //         uint32 count;
    //         archive >> count;

    //         components.resize(prev_count + count);
    //         for (uint32 i = 0; i < count; ++i) {
    //             components[prev_count + i].Serialize(archive, seri);
    //         }

    //         entities.resize(prev_count + count);
    //         for (uint32 i = 0; i < count; ++i) {
    //             Entity entity;
    //             SerializeEntity(archive, entity, seri);
    //             entities[prev_count + i] = entity;
    //             lookup[entity]           = prev_count + i;
    //         }
    //     } else {
    //         archive << components.size();
    //         for (Component& component : components) {
    //             component.Serialize(archive, seri);
    //         }
    //         for (Entity entity : entities) {
    //             SerializeEntity(archive, entity, seri);
    //         }
    //     }
    // }

    // //Read one single component onto an archive, make sure entity are serialized first
    // inline void Component_Serialize(Entity entity, wi::Archive& archive, EntitySerializer& seri)
    // {
    // 	if(archive.IsReadMode())
    // 	{
    // 		bool component_exists;
    // 		archive >> component_exists;
    // 		if (component_exists)
    // 		{
    // 			auto& component = this->Create(entity);
    // 			component.Serialize(archive, seri);
    // 		}
    // 	}
    // 	else
    // 	{
    // 		auto component = this->GetComponent(entity);
    // 		if (component != nullptr)
    // 		{
    // 			archive << true;
    // 			component->Serialize(archive, seri);
    // 		}
    // 		else
    // 		{
    // 			archive << false;
    // 		}
    // 	}
    // }

    // Remove a component of a certain entity if it exists
    inline void Remove(Entity _entity) override {
        auto it = lookup.find(_entity);
        if (it != lookup.end()) {
            // Directly index into components and entities array:
            const auto index  = it->second;
            const auto entity = entities[index];

            if (index < components.size() - 1) {
                // Swap out the dead element with the last one:
                components[index] = std::move(components.back()); // try to use move instead of copy
                entities[index]   = entities.back();

                // Update the lookup table:
                lookup[entities[index]] = index;
            }

            // Shrink the container:
            components.pop_back();
            entities.pop_back();
            lookup.erase(entity);
        }
    }

    // Remove a component of a certain entity if it exists while keeping the current ordering
    inline void RemoveKeepSorted(Entity _entity) override {
        auto it = lookup.find(_entity);
        if (it != lookup.end()) {
            // Directly index into components and entities array:
            const auto index  = it->second;
            const auto entity = entities[index];

            if (index < components.size() - 1) {
                // Move every component left by one that is after this element:
                for (auto i = index + 1; i < components.size(); ++i) {
                    components[i - 1] = std::move(components[i]);
                }
                // Move every entity left by one that is after this element and update lut:
                for (auto i = index + 1; i < entities.size(); ++i) {
                    entities[i - 1]         = entities[i];
                    lookup[entities[i - 1]] = i - 1;
                }
            }

            // Shrink the container:
            components.pop_back();
            entities.pop_back();
            lookup.erase(entity);
        }
    }

    // Place an entity-component to the specified index position while keeping the ordering intact
    inline void MoveItem(Index _index_from, Index _index_to) override {
        assert(_index_from < GetCount());
        assert(_index_to < GetCount());
        if (_index_from == _index_to) {
            return;
        }

        // Save the moved component and entity:
        Component component = std::move(components[_index_from]);
        Entity    entity    = entities[_index_from];

        // Every other entity-component that's in the way gets moved by one and lut is kept updated:
        const int direction = _index_from < _index_to ? 1 : -1;
        for (auto i = _index_from; i != _index_to; i += direction) {
            const auto next     = i + direction;
            components[i]       = std::move(components[next]);
            entities[i]         = entities[next];
            lookup[entities[i]] = i;
        }

        // Saved entity-component moved to the required position:
        components[_index_to] = std::move(component);
        entities[_index_to]   = entity;
        lookup[entity]        = _index_to;
    }

    inline bool Contains(Entity _entity) const override {
        return lookup.contains(_entity);
    }

    // Retrieve component index by entity handle (if not exists, returns ~0ull value)
    inline Index GetIndex(Entity _entity) const override {
        if (lookup.empty())
            return ~0ull;
        const auto it = lookup.find(_entity);
        if (it != lookup.end()) {
            return it->second;
        }
        return ~0ull;
    }

    inline Index GetCount() const override {
        return components.size();
    }

    // Directly index a specific component without indirection
    //	0 <= index < GetCount()
    inline Entity GetEntity(Index _index) const override {
        return entities[_index];
    }

    // Returns the tightly packed [read only] entity array
    inline const Array<Entity>& GetEntityArray() const override {
        return entities;
    }

public:
    // Create a new component and retrieve a reference to it
    inline Component& Create(Entity _entity) {
        // INVALID_ENTITY is not allowed!
        assert(_entity != invalid_entity);

        // Only one of this component type per entity is allowed!
        assert(!lookup.contains(_entity));

        // Entity count must always be the same as the number of coponents!
        assert(entities.size() == components.size());
        assert(lookup.size() == components.size());

        // Update the entity lookup table:
        lookup[_entity] = components.size();

        // New components are always pushed to the end:
        components.emplace_back();

        // Also push corresponding entity:
        entities.push_back(_entity);

        return components.back();
    }

    // Retrieve a [read/write] component specified by an entity (if it exists, otherwise nullptr)
    inline Component* GetComponent(Entity _entity) {
        if (lookup.empty())
            return nullptr;
        auto it = lookup.find(_entity);
        if (it != lookup.end()) {
            return &components[it->second];
        }
        return nullptr;
    }

    // Retrieve a [read only] component specified by an entity (if it exists, otherwise nullptr)
    inline const Component* GetComponent(Entity _entity) const {
        if (lookup.empty())
            return nullptr;
        const auto it = lookup.find(_entity);
        if (it != lookup.end()) {
            return &components[it->second];
        }
        return nullptr;
    }

    // Directly index a specific [read/write] component without indirection
    //	0 <= index < GetCount()
    inline Component& operator[](Index _index) {
        return components[_index];
    }

    // Directly index a specific [read only] component without indirection
    //	0 <= index < GetCount()
    inline const Component& operator[](Index _index) const {
        return components[_index];
    }

    // Returns the tightly packed [read only] component array
    inline const Array<Component>& GetComponentArray() const {
        return components;
    }

private:
    Array<Component> components;
    Array<Entity>    entities;

    std::unordered_map<Entity, Index> lookup;

private:
    // not copyable
    ComponentManager(ComponentManager const&)            = delete;
    ComponentManager& operator=(ComponentManager const&) = delete;

    // Perform deep copy of all the contents of "other" into this
    inline void Copy(const ComponentManager& _rhs) {
        Reserve(GetCount() + _rhs.GetCount());

        for (auto i = 0; i < _rhs.GetCount(); ++i) {
            auto& entity = _rhs.entities[i];
            assert(!Contains(entity));

            entities.push_back(entity);
            lookup[entity] = components.size();
            components.push_back(_rhs.components[i]);
        }
    }

    // Merge in an other component manager of the same type to this.
    //	The other component manager MUST NOT contain any of the same entities!
    //	The other component manager is not retained after this operation!
    inline void Merge(ComponentManager<Component>& _rhs) {
        Reserve(GetCount() + _rhs.GetCount());

        for (auto i = 0; i < _rhs.GetCount(); ++i) {
            Entity entity = _rhs.entities[i];
            assert(!Contains(entity));
            entities.push_back(entity);
            lookup[entity] = components.size();
            components.push_back(std::move(_rhs.components[i]));
        }

        _rhs.Clear();
    }

    inline void Reserve(uint32 _size) {
        components.reserve(_size);
        entities.reserve(_size);
        lookup.reserve(_size);
    }
};

// This is the class to store all component managers,
// this is useful for bulk operation of all attached components within an entity
class ComponentLibrary {
public:
    ComponentLibrary() noexcept  = default;
    ~ComponentLibrary() noexcept = default;

    struct LibraryEntry {
        UniquePtr<IComponentManager> component_manager;
        uint64                       version = 0;
    };
    UnorderedMap<std::string, LibraryEntry> entries;

    // Create an instance of ComponentManager of a certain data type
    //	The name must be unique, it will be used in serialization
    //	version is optional, it will be propagated to ComponentManager::Serialize() inside the EntitySerializer parameter
    template<typename T>
    inline ComponentManager<T>& Register(const std::string& _name, uint64 _version = 0) {
        entries[_name].component_manager = std::move(MakeUnique<ComponentManager<T>>());
        entries[_name].version           = _version;
        return static_cast<ComponentManager<T>&>(*entries[_name].component_manager);
    }

    template<typename T>
    inline ComponentManager<T>* Get(const std::string& _name) {
        auto it = entries.find(_name);
        if (it == entries.end())
            return nullptr;
        return static_cast<ComponentManager<T>*>(it->second.component_manager.get());
    }

    template<typename T>
    inline const ComponentManager<T>* Get(const std::string& _name) const {
        auto it = entries.find(_name);
        if (it == entries.end())
            return nullptr;
        return static_cast<const ComponentManager<T>*>(it->second.component_manager.get());
    }

    inline uint64_t GetVersion(std::string _name) const {
        auto it = entries.find(_name);
        if (it == entries.end())
            return 0;
        return it->second.version;
    }

    // // Serialize all registered component managers
    // inline void Serialize(wi::Archive& archive, EntitySerializer& seri) {
    //     seri.componentlibrary = this;
    //     if (archive.IsReadMode()) {
    //         bool   has_next = false;
    //         size_t begin    = archive.GetPos();

    //         // First pass, gather component type versions and jump over all data:
    //         //	This is so that we can look up other component versions within component serialization if needed
    //         do {
    //             archive >> has_next;
    //             if (has_next) {
    //                 std::string name;
    //                 archive >> name;
    //                 uint64_t jump_pos = 0;
    //                 archive >> jump_pos;
    //                 auto it = entries.find(name);
    //                 if (it != entries.end()) {
    //                     archive >> seri.version;
    //                     seri.library_versions[name] = seri.version;
    //                 }
    //                 archive.Jump(jump_pos);
    //             }
    //         } while (has_next);

    //         // Jump back to beginning of component library data
    //         archive.Jump(begin);

    //         // Second pass, read all component data:
    //         //	At this point, all existing component type versions are available
    //         do {
    //             archive >> has_next;
    //             if (has_next) {
    //                 std::string name;
    //                 archive >> name;
    //                 uint64_t jump_pos = 0;
    //                 archive >> jump_pos;
    //                 auto it = entries.find(name);
    //                 if (it != entries.end()) {
    //                     archive >> seri.version;
    //                     it->second.component_manager->Serialize(archive, seri);
    //                 } else {
    //                     // component manager of this name was not registered, skip serialization by jumping over the data
    //                     archive.Jump(jump_pos);
    //                 }
    //             }
    //         } while (has_next);
    //     } else {
    //         // Save all component type versions:
    //         for (auto& it : entries) {
    //             seri.library_versions[it.first] = it.second.version;
    //         }
    //         // Serialize all component data, at this point component type version lookup is also complete
    //         for (auto& it : entries) {
    //             archive << true;
    //             archive << it.first;                               // name
    //             size_t offset = archive.WriteUnknownJumpPosition();// we will be able to jump from here...
    //             archive << it.second.version;
    //             seri.version = it.second.version;
    //             it.second.component_manager->Serialize(archive, seri);
    //             archive.PatchUnknownJumpPosition(offset);// ...to here, if this component manager was not registered
    //         }
    //         archive << false;
    //     }
    // }

    // // Serialize all components for one entity
    // inline void Entity_Serialize(Entity entity, wi::Archive& archive, EntitySerializer& seri) {
    //     seri.componentlibrary = this;
    //     if (archive.IsReadMode()) {
    //         bool   has_next = false;
    //         size_t begin    = archive.GetPos();

    //         // First pass, gather component type versions and jump over all data:
    //         //	This is so that we can look up other component versions within component serialization if needed
    //         do {
    //             archive >> has_next;
    //             if (has_next) {
    //                 std::string name;
    //                 archive >> name;
    //                 uint64_t jump_pos = 0;
    //                 archive >> jump_pos;
    //                 auto it = entries.find(name);
    //                 if (it != entries.end()) {
    //                     archive >> seri.version;
    //                     seri.library_versions[name] = seri.version;
    //                 }
    //                 archive.Jump(jump_pos);
    //             }
    //         } while (has_next);

    //         // Jump back to beginning of component library data
    //         archive.Jump(begin);

    //         // Second pass: read entities
    //         do {
    //             archive >> has_next;
    //             if (has_next) {
    //                 std::string name;
    //                 archive >> name;
    //                 uint64_t jump_size = 0;
    //                 archive >> jump_size;
    //                 auto it = entries.find(name);
    //                 if (it != entries.end()) {
    //                     archive >> seri.version;
    //                     it->second.component_manager->Component_Serialize(entity, archive, seri);
    //                 } else {
    //                     // component manager of this name was not registered, skip serialization by jumping over the data
    //                     archive.Jump(jump_size);
    //                 }
    //             }
    //         } while (has_next);
    //     } else {
    //         // Save all component type versions:
    //         for (auto& it : entries) {
    //             seri.library_versions[it.first] = it.second.version;
    //         }
    //         // Serialize:
    //         for (auto& it : entries) {
    //             archive << true;
    //             archive << it.first;                               // name
    //             size_t offset = archive.WriteUnknownJumpPosition();// we will be able to jump from here...
    //             archive << it.second.version;
    //             seri.version = it.second.version;
    //             it.second.component_manager->Component_Serialize(entity, archive, seri);
    //             archive.PatchUnknownJumpPosition(offset);// ...to here, if this component manager was not registered
    //         }
    //         archive << false;
    //     }
    // }
private:
    // not copyable
    ComponentLibrary(ComponentLibrary const&) noexcept            = delete;
    ComponentLibrary& operator=(ComponentLibrary const&) noexcept = delete;
};

struct NameComponent {
    std::string name;

    inline void operator=(const std::string& _str) {
        name = _str;
    }
    inline void operator=(std::string&& _str) {
        name = std::move(_str);
    }
    inline bool operator==(const std::string& _str) const {
        return name.compare(_str) == 0;
    }

    // void Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri);
};

struct LayerComponent {
    uint32_t layer_mask = ~0u;

    // Non-serialized attributes:
    uint32_t propagation_mask = ~0u; // This shouldn't be modified by user usually

    constexpr uint32_t GetLayerMask() const {
        return layer_mask & propagation_mask;
    }

    // void Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri);
};

struct HierarchyComponent {
    ECS::Entity parent_id = ECS::invalid_entity;
    uint32_t    layer_mask_bind; // saved child layermask at the time of binding

    // void Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri);
};

}; // namespace Moer::ECS